#include "Optimization/TessellationLowering.hpp"
#include "Optimization/DeadCodeEliminator.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ShaderRecompiler {
namespace {

constexpr std::int64_t PatchIndexCount = 256;
constexpr std::int64_t LocalVertexIndexCount = 8192;
constexpr std::int64_t AddressLimit = 0xffffffffll;

// The bytes a patch's factors take in the factor ring: the outer factors, then the inner ones
// (four and two for quads, three and one for triangles, two outer ones for isolines).
std::int64_t FactorBytes(std::uint32_t domain) {
    return domain == 2u ? 24 : domain == 1u ? 16 : 8;
}

struct Affine {
    std::int64_t vertex = 0;
    std::int64_t relative = 0;
    std::int64_t constant = 0;
    // An added term computed from data (a control-point index loaded from memory): unknown, but
    // independent of the relative patch or vertex index. Only hull LDS reads accept one.
    bool dynamic = false;
    // The dynamic term is dynamicScale * dynamicRoot, the root the value that is not affine (index
    // << 5 is 32 * index; (t << 1) + t of t = index << 3 is 24 * index). Terms of different roots
    // add up to an unknown root and the scales' greatest common divisor. 0 without a dynamic term;
    // a hull read accepts one only in whole control points.
    std::int64_t dynamicScale = 0;
    const IrValue* dynamicRoot = nullptr;

    [[nodiscard]] bool Scalar() const { return vertex == 0 && relative == 0 && !dynamic; }
    bool operator==(const Affine& other) const = default;
};

// Two evaluations that are provably the same value: a dynamic term is never known to equal another.
bool Same(const Affine& left, const Affine& right) {
    return !left.dynamic && !right.dynamic && left == right;
}

std::optional<Affine> CommonMultiple(const Affine& left, const Affine& right) {
    if (left.vertex != right.vertex || left.relative != right.relative || left.constant != right.constant) return std::nullopt;
    const auto multiple = std::gcd(left.dynamic ? left.dynamicScale : 0, right.dynamic ? right.dynamicScale : 0);
    if (multiple == 0) return std::nullopt;
    return Affine{left.vertex, left.relative, left.constant, true, multiple, nullptr};
}

[[noreturn]] void Fail(const char* reason, std::uint32_t pc) {
    char text[256];
    std::snprintf(text, sizeof(text), "tessellation lowering: %s at pc 0x%x", reason, pc);
    throw std::runtime_error(text);
}

bool IsImmediate(const IrValue* raw, std::uint32_t expected) {
    const IrValue* value = raw->Resolve();
    return value->HasImmediate() && value->Type() == IrType::U32 && value->ImmediateU32() == expected;
}

bool IsBase(const IrValue* raw, TessellationBaseKind kind) {
    const IrValue* value = raw->Resolve();
    return value->Opcode() == IrOpcode::TessellationBase && IsImmediate(value->Argument(0), static_cast<std::uint32_t>(kind));
}

bool OwnPatchRead(const IrValue* value) {
    if (SharedAccessOf(value->Opcode()) == SharedAccess::Read) return true;
    return BufferAccessOf(value->Opcode()) == BufferAccess::Read && value->ArgumentCount() > 3u && IsBase(value->Argument(3), TessellationBaseKind::OffChip);
}

std::int64_t FloorDiv(std::int64_t value, std::int64_t divisor) {
    const auto quotient = value / divisor;
    return quotient * divisor > value ? quotient - 1 : quotient;
}

IrValue* VariableOperand(IrValue* value, std::uint32_t constant) {
    if (IsImmediate(value->Argument(1), constant)) return value->Argument(0)->Resolve();
    if (IsImmediate(value->Argument(0), constant)) return value->Argument(1)->Resolve();
    return nullptr;
}

bool IsLaneId(const IrValue* value) {
    return value != nullptr && value->Resolve()->Opcode() == IrOpcode::LaneId;
}

bool IsTrue(const IrValue* value) {
    return value != nullptr && value->HasImmediate() && value->Type() == IrType::Bool && value->ImmediateBool();
}

bool SameBool(IrValue* leftRaw, IrValue* rightRaw) {
    IrValue* left = leftRaw->Resolve();
    IrValue* right = rightRaw->Resolve();
    if (left == right) return true;
    return left->HasImmediate() && right->HasImmediate() && left->Type() == IrType::Bool && right->Type() == IrType::Bool && left->ImmediateBool() == right->ImmediateBool();
}

// A lane-mask word built from words 'word' of ballots with bitwise and, or and not, selects on a
// boolean and all-zero or all-one words. 'otherRaw', when given, must be the same expression of
// the ballots' words 'word + 1' (the upper 32 lanes of a 64-lane mask).
bool MaskWord(IrValue* raw, std::uint32_t word, IrValue* otherRaw) {
    IrValue* value = raw->Resolve();
    IrValue* other = otherRaw != nullptr ? otherRaw->Resolve() : nullptr;
    const auto pair = [&](std::size_t index) { return other != nullptr ? other->Argument(index) : nullptr; };
    if (value->HasImmediate()) {
        if (value->Type() != IrType::U32 || (value->ImmediateU32() != 0u && value->ImmediateU32() != ~0u)) return false;
        return other == nullptr || (other->HasImmediate() && other->Type() == IrType::U32 && other->ImmediateU32() == value->ImmediateU32());
    }
    if (other != nullptr && (other->HasImmediate() || other->Opcode() != value->Opcode())) return false;
    switch (value->Opcode()) {
    case IrOpcode::CompositeExtractU32x4: {
        IrValue* ballot = value->Argument(0)->Resolve();
        if (!IsImmediate(value->Argument(1), word) || ballot->Opcode() != IrOpcode::Ballot) return false;
        if (other == nullptr) return true;
        IrValue* otherBallot = other->Argument(0)->Resolve();
        return IsImmediate(other->Argument(1), word + 1u) && otherBallot->Opcode() == IrOpcode::Ballot && SameBool(ballot->Argument(0), otherBallot->Argument(0));
    }
    case IrOpcode::BitwiseAnd32:
    case IrOpcode::BitwiseOr32:
        return MaskWord(value->Argument(0), word, pair(0)) && MaskWord(value->Argument(1), word, pair(1));
    case IrOpcode::BitwiseXor32:
        return IsImmediate(value->Argument(1), ~0u) && (other == nullptr || IsImmediate(other->Argument(1), ~0u)) && MaskWord(value->Argument(0), word, pair(0));
    case IrOpcode::BitwiseNot32:
        return MaskWord(value->Argument(0), word, pair(0));
    case IrOpcode::SelectU32:
        return value->Argument(0)->Resolve()->Type() == IrType::Bool && (other == nullptr || SameBool(value->Argument(0), other->Argument(0))) && MaskWord(value->Argument(1), word, pair(1)) && MaskWord(value->Argument(2), word, pair(2));
    default:
        return false;
    }
}

// A lane's test of its own bit of a lane mask, (mask >> (lane & 31)) & 1 != 0, the word chosen by
// lane < 32 for a 64-lane mask: the saved exec of s_and_saveexec / s_mov exec, vcc, a mask copied
// by s_cselect_b64 exec and read by v_cndmask, or the entry exec of s_mov_b64 exec, -1.
bool LaneBitTest(IrValue* value, IrValue*& low, IrValue*& high) {
    if (value->Opcode() != IrOpcode::INotEqual32) return false;
    IrValue* masked = VariableOperand(value, 0u);
    if (masked == nullptr || masked->Opcode() != IrOpcode::BitwiseAnd32) return false;
    IrValue* shifted = VariableOperand(masked, 1u);
    if (shifted == nullptr || shifted->Opcode() != IrOpcode::ShiftRightLogical32) return false;
    IrValue* bit = shifted->Argument(1)->Resolve();
    if (bit->Opcode() != IrOpcode::BitwiseAnd32 || !IsLaneId(VariableOperand(bit, 31u))) return false;
    IrValue* word = shifted->Argument(0)->Resolve();
    if (word->Opcode() == IrOpcode::SelectU32) {
        IrValue* lowLanes = word->Argument(0)->Resolve();
        if (lowLanes->Opcode() == IrOpcode::ULessThan32 && IsLaneId(lowLanes->Argument(0)) && IsImmediate(lowLanes->Argument(1), 32u)) {
            low = word->Argument(1)->Resolve();
            high = word->Argument(2)->Resolve();
            return MaskWord(low, 0u, high);
        }
    }
    low = word;
    high = nullptr;
    return MaskWord(low, 0u, nullptr);
}

// (1 << (lane & 31)) - 1
bool IsLowLaneMask(IrValue* raw) {
    IrValue* value = raw->Resolve();
    if (value->Opcode() != IrOpcode::ISub32 || !IsImmediate(value->Argument(1), 1u)) return false;
    IrValue* shifted = value->Argument(0)->Resolve();
    if (shifted->Opcode() != IrOpcode::ShiftLeftLogical32 || !IsImmediate(shifted->Argument(0), 1u)) return false;
    IrValue* bit = shifted->Argument(1)->Resolve();
    return bit->Opcode() == IrOpcode::BitwiseAnd32 && IsLaneId(VariableOperand(bit, 31u));
}

bool IsUpperLane(IrValue* raw) {
    IrValue* value = raw->Resolve();
    return value->Opcode() == IrOpcode::UGreaterThanEqual32 && IsLaneId(value->Argument(0)) && IsImmediate(value->Argument(1), 32u);
}

// v_mbcnt_lo/hi of an all-ones mask: the lane's index.
bool IsLaneCount(IrValue* raw) {
    const auto low = [](IrValue* partRaw) {
        IrValue* part = partRaw->Resolve();
        if (part->Opcode() != IrOpcode::BitCount32) return false;
        IrValue* mask = part->Argument(0)->Resolve();
        if (IsLowLaneMask(mask)) return true;
        return mask->Opcode() == IrOpcode::SelectU32 && IsUpperLane(mask->Argument(0)) && IsImmediate(mask->Argument(1), ~0u) && IsLowLaneMask(mask->Argument(2));
    };
    const auto high = [](IrValue* partRaw) {
        IrValue* part = partRaw->Resolve();
        if (part->Opcode() != IrOpcode::BitCount32) return false;
        IrValue* mask = part->Argument(0)->Resolve();
        return mask->Opcode() == IrOpcode::SelectU32 && IsUpperLane(mask->Argument(0)) && IsLowLaneMask(mask->Argument(1)) && IsImmediate(mask->Argument(2), 0u);
    };
    IrValue* value = raw->Resolve();
    if (low(value)) return true;
    return value->Opcode() == IrOpcode::IAdd32 && ((low(value->Argument(0)) && high(value->Argument(1))) || (low(value->Argument(1)) && high(value->Argument(0))));
}

// The exec predicate a memory, tessellation-interface or export instruction takes last.
IrValue* ExecutionPredicate(IrValue* user) {
    const auto opcode = user->Opcode();
    const bool predicated = SharedAccessOf(opcode) != SharedAccess::None || BufferAccessOf(opcode) != BufferAccess::None || AddressOpcodeInfoOf(opcode).access != AddressAccess::None ||
        ImageOpcodeInfoOf(opcode).access != ImageAccess::None || opcode == IrOpcode::GetTessellationAttribute || opcode == IrOpcode::SetTessellationAttribute || opcode == IrOpcode::SetAttribute;
    if (!predicated || user->ArgumentCount() == 0) return nullptr;
    IrValue* last = user->Argument(user->ArgumentCount() - 1u)->Resolve();
    return last->Type() == IrType::Bool ? last : nullptr;
}

// Matches values by SSA name. Through a loop-carried phi one name stands for another iteration's
// value, so a select reached over a back edge is pruned by the current iteration's predicate. That
// is sound only because a hull loop's exec predicate never widens from one iteration to the next
// (the ffbl / s_cbranch_execnz loops clear lanes as they finish), so the current predicate implies
// the earlier ones; a loop that turns lanes back on would need another argument.
bool Implies(IrValue* rawPredicate, IrValue* rawCondition) {
    IrValue* predicate = rawPredicate->Resolve();
    IrValue* condition = rawCondition->Resolve();
    if (predicate == condition) return true;
    if (IsTrue(condition)) return true;
    return predicate->Opcode() == IrOpcode::LogicalAnd && (Implies(predicate->Argument(0), condition) || Implies(predicate->Argument(1), condition));
}

class AddressEvaluator {
public:
    AddressEvaluator(std::uint32_t vertexCount, std::int64_t relativeCount) : vertexCount(vertexCount), relativeCount(relativeCount) {}

    // allowDynamic: a subexpression that is not affine but does not depend on the relative index
    // becomes a dynamic term instead of failing the evaluation.
    std::optional<Affine> Evaluate(IrValue* value, IrValue* predicate, bool allowDynamic = false) {
        strictMemo.clear();
        dynamicMemo.clear();
        independence.clear();
        active = predicate;
        return evaluate(value, allowDynamic);
    }

    bool IsInvocation(IrValue* value) {
        const auto address = Evaluate(value, nullptr);
        return address && *address == Affine{1, 0, 0, false};
    }

    // Whether a value is computed without the relative index where 'predicate' holds.
    bool IndependentOfRelative(IrValue* value, IrValue* predicate) {
        strictMemo.clear();
        dynamicMemo.clear();
        independence.clear();
        active = predicate;
        return independentOfRelative(value);
    }

    [[nodiscard]] std::int64_t Lowest(const Affine& value) const {
        return value.constant + std::min<std::int64_t>(0, value.vertex * lastVertex()) + std::min<std::int64_t>(0, value.relative * (relativeCount - 1));
    }

    [[nodiscard]] std::int64_t Highest(const Affine& value) const {
        return value.constant + std::max<std::int64_t>(0, value.vertex * lastVertex()) + std::max<std::int64_t>(0, value.relative * (relativeCount - 1));
    }

private:
    [[nodiscard]] std::int64_t lastVertex() const { return vertexCount == 0 ? 0 : static_cast<std::int64_t>(vertexCount) - 1; }

    [[nodiscard]] std::optional<Affine> bounded(const Affine& value) const {
        if (value.dynamic) return value;
        if (Lowest(value) < 0 || Highest(value) > AddressLimit) return std::nullopt;
        return value;
    }

    [[nodiscard]] std::optional<Affine> scale(const Affine& value, std::int64_t factor) const {
        if (factor < 0 || factor >= (1ll << 31) || std::abs(value.dynamicScale) * factor > AddressLimit) return std::nullopt;
        if (value.dynamic && factor == 0) return Affine{};
        return bounded({value.vertex * factor, value.relative * factor, value.constant * factor, value.dynamic, value.dynamicScale * factor, value.dynamicRoot});
    }

    [[nodiscard]] std::optional<std::pair<Affine, Affine>> split(const Affine& value, std::int64_t divisor) const {
        if (value.dynamic) return std::nullopt;
        Affine high;
        Affine low;
        (value.vertex % divisor == 0 ? high.vertex : low.vertex) = value.vertex;
        (value.relative % divisor == 0 ? high.relative : low.relative) = value.relative;
        high.constant = FloorDiv(value.constant, divisor) * divisor;
        low.constant = value.constant - high.constant;
        if (Lowest(low) < 0 || Highest(low) >= divisor) return std::nullopt;
        return std::pair{high, low};
    }

    [[nodiscard]] std::optional<Affine> shiftRight(const Affine& value, std::int64_t shift) const {
        if (shift < 0 || shift >= 32) return std::nullopt;
        const auto divisor = 1ll << shift;
        const auto parts = split(value, divisor);
        if (!parts) return std::nullopt;
        return bounded({parts->first.vertex / divisor, parts->first.relative / divisor, parts->first.constant / divisor});
    }

    [[nodiscard]] std::optional<Affine> mask(const Affine& value, std::int64_t bits) const {
        if (value.dynamic) return std::nullopt;
        if (value.Scalar()) return Affine{0, 0, value.constant & bits};
        const auto width = std::bit_width(static_cast<std::uint64_t>(bits));
        if (width >= 32) return bits == AddressLimit ? std::optional(value) : std::nullopt;
        const auto divisor = 1ll << width;
        const auto parts = split(value, divisor);
        if (!parts) return std::nullopt;
        if (parts->second.Scalar()) return Affine{0, 0, parts->second.constant & bits};
        return bits == divisor - 1 ? std::optional(parts->second) : std::nullopt;
    }

    [[nodiscard]] std::optional<Affine> disjoint(const Affine& high, const Affine& low) const {
        if (high.dynamic || low.dynamic || Lowest(low) < 0) return std::nullopt;
        const auto divisor = 1ll << std::bit_width(static_cast<std::uint64_t>(Highest(low)));
        if (high.vertex % divisor != 0 || high.relative % divisor != 0 || high.constant % divisor != 0) return std::nullopt;
        return bounded({high.vertex + low.vertex, high.relative + low.relative, high.constant + low.constant});
    }

    // Whether a value is computed without the relative index: the search stops at values that are
    // affine without it (the invocation extracted from v1, which also holds the relative index).
    bool independentOfRelative(IrValue* raw) {
        IrValue* start = raw->Resolve();
        if (const auto found = independence.find(start); found != independence.end()) return found->second;
        std::vector<IrValue*> pending{start};
        std::unordered_set<IrValue*> visited;
        bool independent = true;
        while (independent && !pending.empty()) {
            IrValue* value = pending.back();
            pending.pop_back();
            if (value == nullptr || value->HasImmediate() || !visited.insert(value).second) continue;
            if (IsBase(value, TessellationBaseKind::RelativeIndex)) {
                independent = false;
                break;
            }
            if (value != start) {
                if (const auto known = evaluate(value, false)) {
                    independent = known->relative == 0;
                    continue;
                }
            }
            // Where the access predicate holds, a select on a condition it implies takes its first arm
            // (the other arm is the register's value before an exec-masked write).
            if (active != nullptr && value->Opcode() == IrOpcode::SelectU32 && Implies(active, value->Argument(0))) {
                pending.push_back(value->Argument(1)->Resolve());
                continue;
            }
            if (value != start && OwnPatchRead(value)) continue;
            for (std::size_t index = 0; index < value->ArgumentCount(); ++index) {
                if (IrValue* argument = value->Argument(index)) pending.push_back(argument->Resolve());
            }
        }
        independence[start] = independent;
        return independent;
    }

    std::optional<Affine> evaluate(IrValue* raw, bool dynamic) {
        IrValue* value = raw->Resolve();
        if (value->HasImmediate()) {
            if (value->Type() != IrType::U32) return std::nullopt;
            return Affine{0, 0, value->ImmediateU32()};
        }
        auto& memo = dynamic ? dynamicMemo : strictMemo;
        if (const auto found = memo.find(value); found != memo.end()) return found->second;
        memo.emplace(value, std::nullopt);
        auto result = compute(value, dynamic);
        if (!result && dynamic && independentOfRelative(value)) result = Affine{0, 0, 0, true, 1, value};
        memo[value] = result;
        return result;
    }

    std::optional<Affine> compute(IrValue* value, bool dynamic) {
        const auto argument = [&](std::size_t index) { return evaluate(value->Argument(index), dynamic); };
        const auto scalar = [&](std::size_t index) -> std::optional<std::int64_t> {
            const auto operand = argument(index);
            if (!operand || !operand->Scalar()) return std::nullopt;
            return operand->constant;
        };
        switch (value->Opcode()) {
        case IrOpcode::TessellationBase:
            return IsBase(value, TessellationBaseKind::RelativeIndex) ? std::optional(Affine{0, 1, 0}) : std::nullopt;
        case IrOpcode::GetBuiltin:
            return vertexCount != 0 && IsImmediate(value->Argument(0), static_cast<std::uint32_t>(StageInputKind::InvocationId)) ? std::optional(Affine{1, 0, 0}) : std::nullopt;
        case IrOpcode::IAdd32:
        case IrOpcode::ISub32: {
            const auto lhs = argument(0);
            const auto rhs = argument(1);
            if (!lhs || !rhs) return std::nullopt;
            const std::int64_t sign = value->Opcode() == IrOpcode::IAdd32 ? 1 : -1;
            Affine sum{lhs->vertex + sign * rhs->vertex, lhs->relative + sign * rhs->relative, lhs->constant + sign * rhs->constant, lhs->dynamic || rhs->dynamic, lhs->dynamicScale, lhs->dynamicRoot};
            if (rhs->dynamic && !lhs->dynamic) {
                sum.dynamicScale = sign * rhs->dynamicScale;
                sum.dynamicRoot = rhs->dynamicRoot;
            } else if (rhs->dynamic && lhs->dynamicRoot != nullptr && lhs->dynamicRoot == rhs->dynamicRoot) {
                sum.dynamicScale = lhs->dynamicScale + sign * rhs->dynamicScale;
                if (sum.dynamicScale == 0) {
                    sum.dynamic = false;
                    sum.dynamicRoot = nullptr;
                }
            } else if (rhs->dynamic) {
                sum.dynamicScale = std::gcd(lhs->dynamicScale, rhs->dynamicScale);
                sum.dynamicRoot = nullptr;
            }
            return bounded(sum);
        }
        case IrOpcode::IMul32: {
            const auto lhs = argument(0);
            const auto rhs = argument(1);
            if (!lhs || !rhs) return std::nullopt;
            if (rhs->Scalar()) return scale(*lhs, rhs->constant);
            if (lhs->Scalar()) return scale(*rhs, lhs->constant);
            return std::nullopt;
        }
        case IrOpcode::ShiftLeftLogical32: {
            const auto lhs = argument(0);
            const auto shift = scalar(1);
            if (!lhs || !shift || *shift < 0 || *shift >= 31) return std::nullopt;
            return scale(*lhs, 1ll << *shift);
        }
        case IrOpcode::ShiftRightLogical32: {
            const auto lhs = argument(0);
            const auto shift = scalar(1);
            if (!lhs || !shift) return std::nullopt;
            return shiftRight(*lhs, *shift);
        }
        case IrOpcode::BitwiseAnd32: {
            const auto lhs = argument(0);
            const auto rhs = argument(1);
            if (!lhs || !rhs) return std::nullopt;
            if (rhs->Scalar()) return mask(*lhs, rhs->constant);
            if (lhs->Scalar()) return mask(*rhs, lhs->constant);
            return std::nullopt;
        }
        case IrOpcode::BitwiseOr32: {
            const auto lhs = argument(0);
            const auto rhs = argument(1);
            if (!lhs || !rhs) return std::nullopt;
            if (const auto result = disjoint(*lhs, *rhs)) return result;
            return disjoint(*rhs, *lhs);
        }
        case IrOpcode::BitFieldUExtract: {
            const auto source = argument(0);
            const auto offset = scalar(1);
            const auto count = scalar(2);
            if (!source || !offset || !count || *count < 0) return std::nullopt;
            if (*count == 0) return Affine{};
            const auto shifted = shiftRight(*source, *offset);
            if (!shifted || *count >= 32) return shifted;
            return mask(*shifted, (1ll << *count) - 1);
        }
        case IrOpcode::SelectU32: {
            if (active != nullptr && Implies(active, value->Argument(0))) return argument(1);
            const auto lhs = argument(1);
            const auto rhs = argument(2);
            if (!lhs || !rhs) return std::nullopt;
            if (Same(*lhs, *rhs)) return lhs;
            return dynamic ? CommonMultiple(*lhs, *rhs) : std::nullopt;
        }
        default:
            break;
        }
        if (!value->IsPhi() || value->ArgumentCount() == 0) return std::nullopt;
        auto result = argument(0);
        for (std::size_t index = 1; result && index < value->ArgumentCount(); ++index) {
            const auto other = argument(index);
            if (!other) return std::nullopt;
            if (Same(*result, *other)) continue;
            result = dynamic ? CommonMultiple(*result, *other) : std::nullopt;
        }
        return result && (dynamic || !result->dynamic) ? result : std::nullopt;
    }

    std::uint32_t vertexCount;
    std::int64_t relativeCount;
    IrValue* active = nullptr;
    std::unordered_map<IrValue*, std::optional<Affine>> strictMemo;
    std::unordered_map<IrValue*, std::optional<Affine>> dynamicMemo;
    std::unordered_map<IrValue*, bool> independence;
};

bool FirstInvocationOnly(IrValue* raw, AddressEvaluator& evaluator, std::unordered_set<IrValue*>& path) {
    IrValue* value = raw->Resolve();
    if (value->HasImmediate()) return value->Type() == IrType::Bool && !value->ImmediateBool();
    if (!path.insert(value).second) return true;
    const auto invocation = [&](std::size_t index) { return evaluator.IsInvocation(value->Argument(index)); };
    const auto immediate = [&](std::size_t index, std::uint32_t expected) { return IsImmediate(value->Argument(index), expected); };
    const bool result = [&] {
        switch (value->Opcode()) {
        case IrOpcode::LogicalAnd:
            return FirstInvocationOnly(value->Argument(0), evaluator, path) || FirstInvocationOnly(value->Argument(1), evaluator, path);
        case IrOpcode::ULessThan32:
            return invocation(0) && immediate(1, 1u);
        case IrOpcode::UGreaterThan32:
            return immediate(0, 1u) && invocation(1);
        case IrOpcode::ULessThanEqual32:
            return invocation(0) && immediate(1, 0u);
        case IrOpcode::UGreaterThanEqual32:
            return immediate(0, 0u) && invocation(1);
        case IrOpcode::IEqual32:
            return (invocation(0) && immediate(1, 0u)) || (immediate(0, 0u) && invocation(1));
        default:
            break;
        }
        if (!value->IsPhi() || value->ArgumentCount() == 0) return false;
        for (std::size_t index = 0; index < value->ArgumentCount(); ++index) {
            if (!FirstInvocationOnly(value->Argument(index), evaluator, path)) return false;
        }
        return true;
    }();
    path.erase(value);
    return result;
}

struct Access {
    IrBlock* block = nullptr;
    IrValue* inst = nullptr;
    TessellationAttribute kind = TessellationAttribute::LocalOutput;
    bool write = false;
    std::uint32_t components = 1;
    Affine address;
    IrValue* predicate = nullptr;
    std::uint32_t pc = 0;
    // The guest address operand and the instruction offset, kept for a dynamic address that is
    // rewritten from the code rather than from its evaluation.
    IrValue* addressValue = nullptr;
    std::uint32_t memoryOffset = 0;
};

std::uint32_t ComponentCount(IrOpcode opcode) {
    switch (opcode) {
    case IrOpcode::LoadSharedU32x2:
    case IrOpcode::WriteSharedU32x2:
    case IrOpcode::LoadBufferU32x2:
    case IrOpcode::StoreBufferU32x2:
        return 2u;
    case IrOpcode::LoadSharedU32x3:
    case IrOpcode::WriteSharedU32x3:
    case IrOpcode::LoadBufferU32x3:
    case IrOpcode::StoreBufferU32x3:
        return 3u;
    case IrOpcode::LoadSharedU32x4:
    case IrOpcode::WriteSharedU32x4:
    case IrOpcode::LoadBufferU32x4:
    case IrOpcode::StoreBufferU32x4:
        return 4u;
    case IrOpcode::LoadSharedU32:
    case IrOpcode::WriteSharedU32:
    case IrOpcode::LoadBufferU32:
    case IrOpcode::StoreBufferU32:
        return 1u;
    default:
        return 0u;
    }
}

class Lowering {
public:
    Lowering(IrProgram& program, const ShaderTessellationInputInfo& tessellation)
        : program(program), tessellation(tessellation), stage(program.Resources().stage), evaluator(stage == IrShaderStage::TessellationControl ? tessellation.outputControlPoints : 0u, stage == IrShaderStage::Local ? LocalVertexIndexCount : PatchIndexCount) {}

    void Run() {
        for (IrBlock* block : program.BlockOrder()) {
            for (IrValue* inst : block->Instructions()) {
                IrValue* low = nullptr;
                IrValue* high = nullptr;
                if (inst->HasUses() && LaneBitTest(inst, low, high)) replaceValue(inst, laneBool(block, inst, low));
            }
        }
        if (stage == IrShaderStage::TessellationControl) foldControlPointGuard();
        simplifySelects();
        collect();
        derive();
        for (const auto& access : accesses) rewrite(access);
        if (stage == IrShaderStage::TessellationControl) synchronizeControlPointReads();
        if (stage == IrShaderStage::TessellationEvaluation) removePrimitiveExports();
        constexpr DeadCodeEliminator deadCodeEliminator;
        deadCodeEliminator.Eliminate(program);
        std::vector<IrValue*> relative;
        for (IrBlock* block : program.BlockOrder()) {
            for (IrValue* inst : block->Instructions()) {
                if (stage == IrShaderStage::TessellationControl && inst->Opcode() == IrOpcode::Ballot && inst->HasUses()) Fail("the hull program uses a lane mask other than its own lane's bit", 0u);
                if (inst->Opcode() != IrOpcode::TessellationBase || !inst->HasUses()) continue;
                if (IsBase(inst, TessellationBaseKind::RelativeIndex)) {
                    relative.push_back(inst);
                    continue;
                }
                if (IsBase(inst, TessellationBaseKind::PassthroughPrimitive)) Fail("the primitive VGPR is used other than as the passthrough primitive export", 0u);
                Fail("a tessellation ring base is used outside a lowered ring access", 0u);
            }
        }
        requireRelativeIndependence(relative);
        for (IrValue* base : relative) base->ReplaceAllUsesWith(&constant(0u));
        deadCodeEliminator.Eliminate(program);
        program.Metadata().tessellationLocalStride = localStride;
        program.Metadata().tessellationControlStride = controlStride;
    }

private:
    void collect() {
        const auto& memoryInfo = program.Resources().memoryInfo;
        for (IrBlock* block : program.BlockOrder()) {
            for (IrValue* inst : block->Instructions()) {
                const auto opcode = inst->Opcode();
                const auto shared = SharedAccessOf(opcode);
                const auto buffer = BufferAccessOf(opcode);
                if (shared == SharedAccess::None && buffer == BufferAccess::None) continue;
                const auto flags = inst->Flags<MemoryFlags>();
                const auto& memory = memoryInfo.at(flags.index);
                Access access{block, inst, TessellationAttribute::LocalOutput, false, ComponentCount(opcode), {}, inst->Argument(inst->ArgumentCount() - 1u), flags.pc};
                IrValue* address = nullptr;
                if (shared != SharedAccess::None) {
                    if (memory.kind != ResourceKind::Lds) continue;
                    access.write = shared == SharedAccess::Write;
                    if (access.components == 0u || (shared != SharedAccess::Read && shared != SharedAccess::Write)) Fail("only 32-bit LDS reads and writes are supported in tessellation stages", flags.pc);
                    if (stage == IrShaderStage::Local && access.write) {
                        access.kind = TessellationAttribute::LocalOutput;
                    } else if (stage == IrShaderStage::TessellationControl && !access.write) {
                        access.kind = TessellationAttribute::ControlInput;
                    } else {
                        Fail("unsupported LDS access in a tessellation stage", flags.pc);
                    }
                    address = inst->Argument(0);
                } else {
                    if (memory.kind != ResourceKind::Buffer || (!IsBase(inst->Argument(3), TessellationBaseKind::OffChip) && !IsBase(inst->Argument(3), TessellationBaseKind::Factor))) continue;
                    if (buffer == BufferAccess::Atomic || access.components == 0u || memory.dataBits != 32u || memory.formatted || memory.typed || memory.d16 || memory.idxen) Fail("only untyped 32-bit tessellation ring accesses are supported", flags.pc);
                    access.write = buffer == BufferAccess::Write;
                    const bool factor = IsBase(inst->Argument(3), TessellationBaseKind::Factor);
                    if (stage == IrShaderStage::TessellationControl && access.write) {
                        access.kind = factor ? TessellationAttribute::Factor : TessellationAttribute::ControlOutput;
                    } else if (stage == IrShaderStage::TessellationControl && !factor) {
                        // The hull reading back control points of its patch (its patch-constant phase).
                        access.kind = TessellationAttribute::ControlOutput;
                    } else if (stage == IrShaderStage::TessellationEvaluation && !access.write && !factor) {
                        access.kind = TessellationAttribute::EvaluationInput;
                    } else {
                        Fail("unsupported tessellation ring access", flags.pc);
                    }
                    address = memory.offen ? inst->Argument(2) : nullptr;
                }
                if (address != nullptr) {
                    const auto evaluated = evaluator.Evaluate(address, access.predicate, access.kind == TessellationAttribute::ControlInput || access.kind == TessellationAttribute::EvaluationInput);
                    if (!evaluated) Fail("the tessellation ring or LDS address is not affine in the control point and relative patch", flags.pc);
                    access.address = *evaluated;
                    access.addressValue = address;
                }
                access.memoryOffset = memory.offset;
                access.address.constant += memory.offset;
                if ((access.address.constant & 3) != 0) Fail("unaligned tessellation ring or LDS address", flags.pc);
                accesses.push_back(access);
            }
        }
    }

    void agree(std::uint32_t& stride, std::int64_t value, std::uint32_t pc) const {
        if (value <= 0 || (value & 3) != 0 || value > 0xffff || (stride != 0u && stride != value)) Fail("inconsistent tessellation stride", pc);
        stride = static_cast<std::uint32_t>(value);
    }

    void derive() {
        const auto inputs = static_cast<std::int64_t>(tessellation.inputControlPoints);
        const auto outputs = static_cast<std::int64_t>(tessellation.outputControlPoints);
        for (const auto& access : accesses) {
            if (access.kind == TessellationAttribute::LocalOutput) agree(localStride, access.address.relative, access.pc);
            if (access.kind == TessellationAttribute::ControlInput) {
                if (access.address.vertex != 0 && !access.address.dynamic) agree(localStride, access.address.vertex, access.pc);
                if (access.address.relative <= 0 || access.address.relative % inputs != 0) Fail("the hull shader reads LDS of other patches", access.pc);
                agree(localStride, access.address.relative / inputs, access.pc);
            }
            if (access.kind == TessellationAttribute::ControlOutput && access.write && access.address.vertex != 0) agree(controlStride, access.address.vertex, access.pc);
            if (access.kind == TessellationAttribute::EvaluationInput) {
                if (access.address.relative <= 0 || access.address.relative % outputs != 0) Fail(access.address.dynamic ? "the domain shader reads off-chip memory at a computed offset outside its patch's control points" : "domain shader reads of patch constants or other patches are unsupported", access.pc);
                agree(controlStride, access.address.relative / outputs, access.pc);
            }
        }
        std::optional<std::int64_t> patchStride;
        std::int64_t patchLowest = AddressLimit;
        std::int64_t patchHighest = 0;
        for (auto& access : accesses) {
            const auto& address = access.address;
            const auto end = address.constant + 4 * static_cast<std::int64_t>(access.components);
            switch (access.kind) {
            case TessellationAttribute::LocalOutput:
                if (address.vertex != 0 || address.constant < 0 || end > localStride) Fail("the local stage writes outside its vertex's LDS slot", access.pc);
                break;
            case TessellationAttribute::ControlInput:
                if (localStride == 0u) Fail("the hull shader reads no LDS per control point, so the local stride is unknown", access.pc);
                if (address.relative != inputs * localStride || address.constant < 0) Fail("the hull shader reads LDS outside its patch", access.pc);
                // A computed term must move the read by whole control points (index * stride): any
                // other one (an LDS base from user data) would be clamped as if it were an index.
                if (address.dynamic && (address.dynamicScale == 0 || address.dynamicScale % localStride != 0)) Fail("the hull shader reads LDS at a computed offset that is not a whole number of control points", access.pc);
                // A dynamic control-point index is clamped to the patch where it is emitted.
                if (!address.dynamic && ((address.vertex != 0 && address.vertex != localStride) || end + address.vertex * (outputs - 1) > inputs * localStride)) Fail("the hull shader reads LDS outside its patch", access.pc);
                break;
            case TessellationAttribute::ControlOutput:
                if (!access.write) {
                    if (controlStride == 0u || address.relative != outputs * controlStride || address.constant < 0 || (address.vertex != 0 && address.vertex != controlStride) || end + address.vertex * (outputs - 1) > outputs * controlStride) {
                        Fail("the hull shader reads off-chip memory other than its patch's control points", access.pc);
                    }
                    break;
                }
                if (address.vertex == 0) {
                    if (controlStride == 0u || address.constant < outputs * controlStride) Fail("the hull shader writes off-chip memory outside its control points and patch constants", access.pc);
                    if (patchStride && *patchStride != address.relative) Fail("inconsistent patch-constant stride", access.pc);
                    patchStride = address.relative;
                    patchLowest = std::min(patchLowest, address.constant);
                    patchHighest = std::max(patchHighest, end);
                    access.kind = TessellationAttribute::PatchOutput;
                    requireFirstInvocation(access);
                } else if (address.relative != outputs * controlStride || address.constant < 0 || end > controlStride) {
                    Fail("the hull shader writes off-chip memory outside its control point", access.pc);
                }
                break;
            case TessellationAttribute::EvaluationInput:
                if (address.constant < 0 || end > outputs * controlStride) Fail("the domain shader reads off-chip memory outside its patch's control points", access.pc);
                if (address.dynamic && (address.dynamicScale == 0 || address.dynamicScale % controlStride != 0)) Fail("the domain shader reads off-chip memory at a computed offset that is not a whole number of control points", access.pc);
                break;
            case TessellationAttribute::Factor: {
                const auto bytes = FactorBytes(tessellation.domain);
                if (address.vertex != 0 || address.relative != bytes || address.constant < 0 || end > bytes) Fail("the hull shader writes tessellation factors outside its patch's factors", access.pc);
                requireFirstInvocation(access);
                break;
            }
            case TessellationAttribute::PatchOutput:
                break;
            }
        }
        if (patchStride && (*patchStride == 0 || std::abs(*patchStride) < patchHighest - patchLowest)) Fail("patch-constant records of neighbouring patches overlap", 0u);
    }

    // A relative index may reach a dynamic hull LDS read only with the stride of a patch's inputs,
    // which replacing the index by 0 removes; every other use must not depend on it.
    bool dynamicInputAddress(IrValue* user, IrValue* value) {
        if (user->Opcode() != IrOpcode::GetTessellationAttribute || user->Argument(1)->Resolve() != value->Resolve()) return false;
        const bool control = IsImmediate(user->Argument(0), static_cast<std::uint32_t>(TessellationAttribute::ControlInput));
        const bool evaluation = IsImmediate(user->Argument(0), static_cast<std::uint32_t>(TessellationAttribute::EvaluationInput));
        if (!control && !evaluation) return false;
        const auto address = evaluator.Evaluate(value, user->Argument(2), true);
        const auto stride = control ? static_cast<std::int64_t>(tessellation.inputControlPoints) * localStride : static_cast<std::int64_t>(tessellation.outputControlPoints) * controlStride;
        return address && address->dynamic && address->relative == stride;
    }

    void requireRelativeIndependence(const std::vector<IrValue*>& bases) {
        std::vector<IrValue*> pending(bases.begin(), bases.end());
        std::unordered_set<IrValue*> visited(bases.begin(), bases.end());
        while (!pending.empty()) {
            IrValue* value = pending.back();
            pending.pop_back();
            for (IrValue* user : value->Uses()) {
                switch (user->Opcode()) {
                case IrOpcode::Identity:
                case IrOpcode::IAdd32:
                case IrOpcode::ISub32:
                case IrOpcode::IMul32:
                case IrOpcode::ShiftLeftLogical32:
                case IrOpcode::ShiftRightLogical32:
                case IrOpcode::BitwiseAnd32:
                case IrOpcode::BitwiseOr32:
                case IrOpcode::BitFieldUExtract:
                case IrOpcode::SelectU32:
                case IrOpcode::Phi: {
                    // A value the relative index does not change (the invocation extracted from v1)
                    // ends the walk.
                    const auto known = evaluator.Evaluate(user, nullptr);
                    if (known && known->relative == 0) continue;
                    if (visited.insert(user).second) pending.push_back(user);
                    continue;
                }
                default:
                    break;
                }
                if (dynamicInputAddress(user, value)) continue;
                if (const auto address = evaluator.Evaluate(value, nullptr)) {
                    if (address->relative == 0) continue;
                } else if (evaluator.IndependentOfRelative(value, ExecutionPredicate(user))) {
                    continue;
                }
                Fail("the relative patch or vertex index is used outside a tessellation ring or LDS address", 0u);
            }
        }
    }

    void requireFirstInvocation(const Access& access) {
        std::unordered_set<IrValue*> path;
        if (!FirstInvocationOnly(access.predicate, evaluator, path)) Fail("per-patch tessellation output is not written by control point 0 alone", access.pc);
    }

    IrValue& constant(std::uint32_t value) {
        IrValue& result = program.CreateValue(IrOpcode::Void, IrType::U32);
        result.SetImmediateU32(value);
        return result;
    }

    // Replaces a value everywhere, the blocks' branch conditions included (RAUW leaves them).
    void replaceValue(IrValue* old, IrValue* replacement) {
        old->ReplaceAllUsesWith(replacement);
        for (auto& info : program.Metadata().blockInfo) {
            if (info.condition != nullptr && info.condition->Resolve() == old) info.condition = replacement;
            if (info.indirectTarget != nullptr && info.indirectTarget->Resolve() == old) info.indirectTarget = replacement;
        }
    }

    IrValue& boolean(bool value) {
        IrValue& result = program.CreateValue(IrOpcode::Void, IrType::Bool);
        result.SetImmediateBool(value);
        return result;
    }

    IrValue& insertAt(IrBlock* block, IrValue* position, IrOpcode opcode, std::initializer_list<IrValue*> arguments) {
        IrValue& value = program.CreateValue(opcode, IrOpcodeType(opcode));
        for (IrValue* argument : arguments) value.AddArgument(argument);
        block->InsertInstructionBefore(position, &value);
        return value;
    }

    // The bit a lane reads of a mask word MaskWord accepts: the same expression of the ballots'
    // sources in that lane (on the host a TCS lane index is gl_InvocationID, not its subgroup lane,
    // so the ballot bits themselves would belong to other invocations).
    IrValue* laneBool(IrBlock* block, IrValue* position, IrValue* raw) {
        IrValue* value = raw->Resolve();
        if (value->HasImmediate()) return &boolean(value->ImmediateU32() != 0u);
        switch (value->Opcode()) {
        case IrOpcode::CompositeExtractU32x4:
            return value->Argument(0)->Resolve()->Argument(0)->Resolve();
        case IrOpcode::BitwiseAnd32:
        case IrOpcode::BitwiseOr32: {
            IrValue* lhs = laneBool(block, position, value->Argument(0));
            IrValue* rhs = laneBool(block, position, value->Argument(1));
            return &insertAt(block, position, value->Opcode() == IrOpcode::BitwiseAnd32 ? IrOpcode::LogicalAnd : IrOpcode::LogicalOr, {lhs, rhs});
        }
        case IrOpcode::BitwiseXor32:
        case IrOpcode::BitwiseNot32: {
            IrValue* operand = laneBool(block, position, value->Argument(0));
            return &insertAt(block, position, IrOpcode::LogicalNot, {operand});
        }
        default: {
            IrValue* taken = laneBool(block, position, value->Argument(1));
            IrValue* other = laneBool(block, position, value->Argument(2));
            return &insertAt(block, position, IrOpcode::SelectU1, {value->Argument(0)->Resolve(), taken, other});
        }
        }
    }

    // A host hull invocation is a control point below the output control-point count, which the
    // prologue puts in s3[15:8]: the hull's opening v_cmpx_lt_u32 of its lane index against it holds.
    void foldControlPointGuard() {
        for (IrBlock* block : program.BlockOrder()) {
            for (IrValue* inst : block->Instructions()) {
                if (inst->Opcode() != IrOpcode::ULessThan32 || !inst->HasUses() || !IsLaneCount(inst->Argument(0))) continue;
                const IrValue* bound = inst->Argument(1)->Resolve();
                if (bound->HasImmediate() && bound->Type() == IrType::U32 && bound->ImmediateU32() >= tessellation.outputControlPoints) replaceValue(inst, &boolean(true));
            }
        }
    }

    // Selects and conjunctions on a condition that holds, and a select whose arm selects again on
    // its condition (an exec-masked write of a register that the same exec wrote before).
    void simplifySelects() {
        for (bool changed = true; changed;) {
            changed = false;
            for (IrBlock* block : program.BlockOrder()) {
                for (IrValue* inst : block->Instructions()) {
                    const auto opcode = inst->Opcode();
                    if (opcode == IrOpcode::LogicalAnd) {
                        const bool first = IsTrue(inst->Argument(0)->Resolve());
                        if ((first || IsTrue(inst->Argument(1)->Resolve())) && inst->HasUses()) {
                            replaceValue(inst, inst->Argument(first ? 1u : 0u)->Resolve());
                            changed = true;
                        }
                        continue;
                    }
                    if (opcode != IrOpcode::SelectU32 && opcode != IrOpcode::SelectF32 && opcode != IrOpcode::SelectU1) continue;
                    IrValue* condition = inst->Argument(0)->Resolve();
                    if (IsTrue(condition)) {
                        if (inst->HasUses()) {
                            replaceValue(inst, inst->Argument(1)->Resolve());
                            changed = true;
                        }
                        continue;
                    }
                    for (std::size_t arm = 1; arm <= 2; ++arm) {
                        IrValue* nested = inst->Argument(arm)->Resolve();
                        if (nested != inst && nested->Opcode() == opcode && nested->Argument(0)->Resolve() == condition) {
                            inst->ReplaceArgument(arm, nested->Argument(arm)->Resolve());
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    IrValue& insert(const Access& access, IrOpcode opcode, IrType type, std::initializer_list<IrValue*> arguments) {
        IrValue& value = program.CreateValue(opcode, type);
        for (IrValue* argument : arguments) value.AddArgument(argument);
        access.block->InsertInstructionBefore(access.inst, &value);
        return value;
    }

    IrValue& componentAddress(const Access& access, std::uint32_t component) {
        if (access.address.dynamic) {
            const std::uint32_t byteOffset = access.memoryOffset + 4u * component;
            return byteOffset == 0u ? *access.addressValue : insert(access, IrOpcode::IAdd32, IrType::U32, {access.addressValue, &constant(byteOffset)});
        }
        IrValue& offset = constant(static_cast<std::uint32_t>(access.address.constant) + 4u * component);
        const bool indexed = access.kind == TessellationAttribute::ControlInput || (access.kind == TessellationAttribute::ControlOutput && !access.write);
        if (!indexed || access.address.vertex == 0) return offset;
        IrValue& invocation = insert(access, IrOpcode::GetBuiltin, IrType::U32, {&constant(static_cast<std::uint32_t>(StageInputKind::InvocationId)), &constant(0u)});
        IrValue& scaled = insert(access, IrOpcode::IMul32, IrType::U32, {&invocation, &constant(static_cast<std::uint32_t>(access.address.vertex))});
        return insert(access, IrOpcode::IAdd32, IrType::U32, {&scaled, &offset});
    }

    void keepComputedReadsInPatch(IrValue& base) {
        for (IrBlock* block : program.BlockOrder()) {
            for (IrValue* inst : block->Instructions()) {
                if (inst->Opcode() != IrOpcode::GetTessellationAttribute) continue;
                const bool control = IsImmediate(inst->Argument(0), static_cast<std::uint32_t>(TessellationAttribute::ControlInput));
                if (!control && !IsImmediate(inst->Argument(0), static_cast<std::uint32_t>(TessellationAttribute::EvaluationInput))) continue;
                const auto address = evaluator.Evaluate(inst->Argument(1), inst->Argument(2), true);
                if (!address || !address->dynamic || address->relative == 0) continue;
                IrValue& patch = program.CreateValue(IrOpcode::IMul32, IrType::U32);
                patch.AddArgument(&base);
                patch.AddArgument(&constant(static_cast<std::uint32_t>(address->relative)));
                block->InsertInstructionBefore(inst, &patch);
                IrValue& inPatch = program.CreateValue(IrOpcode::ISub32, IrType::U32);
                inPatch.AddArgument(inst->Argument(1));
                inPatch.AddArgument(&patch);
                block->InsertInstructionBefore(inst, &inPatch);
                inst->ReplaceArgument(1, &inPatch);
            }
        }
    }

    void rewrite(const Access& access) {
        IrValue& kind = constant(static_cast<std::uint32_t>(access.kind));
        const bool shared = SharedAccessOf(access.inst->Opcode()) != SharedAccess::None;
        if (access.write) {
            for (std::uint32_t component = 0; component < access.components; ++component) {
                IrValue* data = nullptr;
                if (shared) {
                    data = access.inst->Argument(1u + component);
                } else if (access.components == 1u) {
                    data = access.inst->Argument(4u);
                } else {
                    const auto extract = access.components == 2u ? IrOpcode::CompositeExtractU32x2 : access.components == 3u ? IrOpcode::CompositeExtractU32x3 : IrOpcode::CompositeExtractU32x4;
                    data = &insert(access, extract, IrType::U32, {access.inst->Argument(4u), &constant(component)});
                }
                static_cast<void>(insert(access, IrOpcode::SetTessellationAttribute, IrType::Void, {&kind, &componentAddress(access, component), data, access.predicate}));
            }
        } else {
            std::vector<IrValue*> values;
            for (std::uint32_t component = 0; component < access.components; ++component) {
                values.push_back(&insert(access, IrOpcode::GetTessellationAttribute, IrType::U32, {&kind, &componentAddress(access, component), access.predicate}));
            }
            IrValue* replacement = values.front();
            if (access.components == 2u) replacement = &insert(access, IrOpcode::CompositeConstructU32x2, IrType::U32x2, {values[0], values[1]});
            if (access.components == 3u) replacement = &insert(access, IrOpcode::CompositeConstructU32x3, IrType::U32x3, {values[0], values[1], values[2]});
            if (access.components == 4u) replacement = &insert(access, IrOpcode::CompositeConstructU32x4, IrType::U32x4, {values[0], values[1], values[2], values[3]});
            access.inst->ReplaceAllUsesWith(replacement);
        }
        access.block->RemoveInstruction(access.inst);
        access.inst->Invalidate();
    }

    // Whether the block writes a control point before 'position' (null: anywhere in it).
    static bool WritesControlPoint(IrBlock* block, const IrValue* position) {
        for (IrValue* inst : block->Instructions()) {
            if (inst == position) return false;
            if (inst->Opcode() == IrOpcode::SetTessellationAttribute && IsImmediate(inst->Argument(0), static_cast<std::uint32_t>(TessellationAttribute::ControlOutput))) return true;
        }
        return false;
    }

    // The header of the selection whose branch arm starts at the block, when the header is the
    // block's only predecessor; otherwise null.
    IrBlock* SelectionHeaderOf(IrBlock* block) {
        const auto& order = program.BlockOrder();
        const auto& infos = program.Metadata().blockInfo;
        if (order.size() != infos.size()) Fail("the hull program's block metadata is inconsistent", 0u);
        const auto position = std::find(order.begin(), order.end(), block);
        if (position == order.end()) return nullptr;
        const auto id = infos[static_cast<std::size_t>(position - order.begin())].id;
        IrBlock* header = nullptr;
        std::size_t predecessors = 0;
        for (std::size_t index = 0; index < infos.size(); ++index) {
            const auto& terminator = infos[index].terminator;
            if (terminator.kind == TerminatorKind::Branch) {
                predecessors += terminator.trueBlock == id ? 1u : 0u;
                continue;
            }
            if (terminator.kind == TerminatorKind::IndirectBranch && std::find(terminator.indirectTargets.begin(), terminator.indirectTargets.end(), id) != terminator.indirectTargets.end()) return nullptr;
            if (terminator.kind != TerminatorKind::ConditionalBranch || (terminator.trueBlock != id && terminator.falseBlock != id)) continue;
            ++predecessors;
            if (terminator.trueBlock == terminator.falseBlock || terminator.loopHeader || terminator.mergeBlock == id || terminator.mergeBlock == InvalidControlFlowId) return nullptr;
            header = order[index];
        }
        return predecessors == 1u ? header : nullptr;
    }

    // A branch condition that folds to a constant (the folded lane guard, !true).
    static std::optional<bool> ConstantCondition(IrValue* raw) {
        if (raw == nullptr) return std::nullopt;
        IrValue* value = raw->Resolve();
        if (value->HasImmediate()) return value->Type() == IrType::Bool ? std::optional(value->ImmediateBool()) : std::nullopt;
        if (value->Opcode() == IrOpcode::LogicalNot) {
            const auto operand = ConstantCondition(value->Argument(0));
            return operand ? std::optional(!*operand) : std::nullopt;
        }
        if (value->Opcode() == IrOpcode::LogicalAnd || value->Opcode() == IrOpcode::LogicalOr) {
            const bool conjunction = value->Opcode() == IrOpcode::LogicalAnd;
            const auto lhs = ConstantCondition(value->Argument(0));
            const auto rhs = ConstantCondition(value->Argument(1));
            if ((lhs && *lhs != conjunction) || (rhs && *rhs != conjunction)) return !conjunction;
            if (lhs && rhs) return conjunction;
        }
        return std::nullopt;
    }

    // Whether every invocation that runs the program runs the block: each branch the block is
    // control dependent on (by post-dominance) has a constant condition, which a wave-mask branch
    // (its wave-any of the lane condition) and a scalar branch take the same way in every invocation.
    bool UniformlyReached(const IrBlock* target) {
        const auto& order = program.BlockOrder();
        const auto& infos = program.Metadata().blockInfo;
        const std::size_t count = order.size();
        if (count == 0 || count != infos.size()) return false;
        std::unordered_map<std::uint32_t, std::size_t> indexOf;
        for (std::size_t index = 0; index < count; ++index) indexOf.emplace(infos[index].id, index);
        // Successor indices; 'count' is a virtual exit after every Return.
        std::vector<std::vector<std::size_t>> successors(count);
        for (std::size_t index = 0; index < count; ++index) {
            const auto& terminator = infos[index].terminator;
            const auto add = [&](std::uint32_t id) {
                const auto found = indexOf.find(id);
                if (found == indexOf.end()) return false;
                if (std::find(successors[index].begin(), successors[index].end(), found->second) == successors[index].end()) successors[index].push_back(found->second);
                return true;
            };
            switch (terminator.kind) {
            case TerminatorKind::Branch:
                if (!add(terminator.trueBlock)) return false;
                break;
            case TerminatorKind::ConditionalBranch:
                if (!add(terminator.trueBlock) || !add(terminator.falseBlock)) return false;
                break;
            case TerminatorKind::IndirectBranch:
                for (const auto id : terminator.indirectTargets) {
                    if (!add(id)) return false;
                }
                break;
            case TerminatorKind::Return:
                successors[index].push_back(count);
                break;
            default:
                return false;
            }
        }
        // Post-dominator sets as bit rows over count + 1 nodes, by iteration to a fixed point.
        const std::size_t words = (count + 1 + 63) / 64;
        std::vector<std::uint64_t> rows((count + 1) * words, ~0ull);
        const auto row = [&](std::size_t node) { return rows.begin() + static_cast<std::ptrdiff_t>(node * words); };
        const auto contains = [&](std::size_t node, std::size_t member) { return ((*(row(node) + static_cast<std::ptrdiff_t>(member / 64)) >> (member % 64)) & 1ull) != 0; };
        std::fill(row(count), row(count) + static_cast<std::ptrdiff_t>(words), 0ull);
        *(row(count) + static_cast<std::ptrdiff_t>(count / 64)) |= 1ull << (count % 64);
        std::vector<std::uint64_t> next(words);
        for (bool changed = true; changed;) {
            changed = false;
            for (std::size_t node = count; node-- > 0;) {
                std::fill(next.begin(), next.end(), ~0ull);
                for (const auto successor : successors[node]) {
                    for (std::size_t word = 0; word < words; ++word) next[word] &= *(row(successor) + static_cast<std::ptrdiff_t>(word));
                }
                next[node / 64] |= 1ull << (node % 64);
                if (!std::equal(next.begin(), next.end(), row(node))) {
                    std::copy(next.begin(), next.end(), row(node));
                    changed = true;
                }
            }
        }
        // Blocks the entry reaches; blocks that reach the exit (the others keep full rows, so the
        // block must reach it for its row and the rows above it to mean post-dominance).
        std::vector<bool> reached(count, false);
        std::vector<std::size_t> pending{0};
        reached[0] = true;
        while (!pending.empty()) {
            const auto node = pending.back();
            pending.pop_back();
            for (const auto successor : successors[node]) {
                if (successor < count && !reached[successor]) {
                    reached[successor] = true;
                    pending.push_back(successor);
                }
            }
        }
        std::vector<bool> exits(count + 1, false);
        exits[count] = true;
        for (bool changed = true; changed;) {
            changed = false;
            for (std::size_t node = 0; node < count; ++node) {
                if (exits[node] || std::none_of(successors[node].begin(), successors[node].end(), [&](std::size_t successor) { return exits[successor]; })) continue;
                exits[node] = true;
                changed = true;
            }
        }
        const auto position = std::find(order.begin(), order.end(), target);
        if (position == order.end()) return false;
        const auto block = static_cast<std::size_t>(position - order.begin());
        if (!exits[block]) return false;
        for (std::size_t header = 0; header < count; ++header) {
            if (!reached[header] || successors[header].size() < 2u) continue;
            // Control dependent: the block post-dominates a successor but not the header strictly.
            if (header != block && contains(header, block)) continue;
            const bool dependent = std::any_of(successors[header].begin(), successors[header].end(), [&](std::size_t successor) { return successor < count && contains(successor, block); });
            if (!dependent) continue;
            if (infos[header].terminator.kind != TerminatorKind::ConditionalBranch || !ConstantCondition(infos[header].condition)) return false;
        }
        return true;
    }

    // The console's hull wave sees its own off-chip stores in order; the host's control-point
    // invocations need a barrier before one of them reads the patch's other control points. Every
    // control point of the patch must run it, so it leaves the selections that hold the read (the
    // hull's invocation-0 region branches on a wave mask, which other host subgroups may not take)
    // and is placed at the end of the outermost such selection's header, never above a
    // control-point write (in block order: one in another arm of the selection stops it too).
    void synchronizeControlPointReads() {
        IrBlock* readBlock = nullptr;
        IrValue* firstRead = nullptr;
        for (IrBlock* block : program.BlockOrder()) {
            for (IrValue* inst : block->Instructions()) {
                const auto opcode = inst->Opcode();
                if ((opcode != IrOpcode::GetTessellationAttribute && opcode != IrOpcode::SetTessellationAttribute) || !IsImmediate(inst->Argument(0), static_cast<std::uint32_t>(TessellationAttribute::ControlOutput))) continue;
                if (opcode == IrOpcode::GetTessellationAttribute) {
                    if (firstRead == nullptr) {
                        firstRead = inst;
                        readBlock = block;
                    }
                } else if (firstRead != nullptr) {
                    Fail("the hull program writes a control point after reading its patch's control points", 0u);
                }
            }
        }
        if (firstRead == nullptr) return;
        const auto& order = program.BlockOrder();
        const auto indexOf = [&](IrBlock* block) { return static_cast<std::size_t>(std::find(order.begin(), order.end(), block) - order.begin()); };
        const auto readIndex = indexOf(readBlock);
        // A control-point write after the block at 'from' and before the first read, in block order.
        const auto writesAfter = [&](std::size_t from) {
            for (std::size_t index = from + 1; index <= readIndex && index < order.size(); ++index) {
                if (WritesControlPoint(order[index], index == readIndex ? firstRead : nullptr)) return true;
            }
            return false;
        };
        IrBlock* block = readBlock;
        IrValue* position = firstRead;
        if (!WritesControlPoint(readBlock, firstRead)) {
            while (IrBlock* header = SelectionHeaderOf(block)) {
                if (writesAfter(indexOf(header))) break;
                block = header;
                position = nullptr;
            }
        }
        // The walk stops at a loop, a merge or a block with several predecessors, and below a
        // control-point write: the place it reaches must still run in every control point.
        if (!UniformlyReached(block)) Fail("cannot place the control-point read-back barrier in uniform control flow", 0u);
        IrValue& barrier = program.CreateValue(IrOpcode::Barrier, IrType::Void);
        if (position != nullptr) {
            block->InsertInstructionBefore(position, &barrier);
        } else {
            block->AppendInstruction(&barrier);
        }
    }

    void removePrimitiveExports() {
        const auto& exports = program.Metadata().exportInfo;
        for (IrBlock* block : program.BlockOrder()) {
            auto& instructions = block->Instructions();
            for (auto it = instructions.begin(); it != instructions.end();) {
                IrValue* inst = *it;
                const auto flags = inst->Flags<ExportFlags>();
                if (inst->Opcode() != IrOpcode::SetAttribute || exports.at(flags.index).kind != ExportTargetKind::Primitive) {
                    ++it;
                    continue;
                }
                const IrValue* data = inst->Argument(0)->Resolve();
                if ((exports.at(flags.index).en & 1u) == 0u || data->Opcode() != IrOpcode::CompositeConstructU32x4 || !IsBase(data->Argument(0), TessellationBaseKind::PassthroughPrimitive)) Fail("the domain shader exports a primitive other than the passthrough primitive", flags.pc);
                it = instructions.erase(it);
                inst->SetParent(nullptr);
                inst->Invalidate();
            }
        }
    }

    IrProgram& program;
    const ShaderTessellationInputInfo& tessellation;
    IrShaderStage stage;
    AddressEvaluator evaluator;
    std::vector<Access> accesses;
    std::uint32_t localStride = 0;
    std::uint32_t controlStride = 0;
};

}

void TessellationLowering::Lower(IrProgram& program, const ShaderTessellationInputInfo& tessellation) const {
    const auto stage = program.Resources().stage;
    if (stage != IrShaderStage::Local && stage != IrShaderStage::TessellationControl && stage != IrShaderStage::TessellationEvaluation) return;
    Lowering(program, tessellation).Run();
}

}

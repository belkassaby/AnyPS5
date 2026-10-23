# GT7 graphics settings: a small dialog that writes graphics-settings.json, which run-gt7.ps1 applies
# (as environment defaults) at every launch. "Save and launch" starts the game right away.
#   powershell -ExecutionPolicy Bypass -File gt7-settings.ps1
param([switch]$NoLaunch)
Set-Location $PSScriptRoot
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$path = Join-Path $PSScriptRoot 'graphics-settings.json'
$presets = [ordered]@{
    'Performance' = @{ resolution = '1080p'; presentMode = 'mailbox'; output120 = $false; precompile = $true; vramGeometry = $true; imagePoolMiB = 2048; textureCacheMiB = 6144; hostImportMiB = 12288; workingSetMiB = 0; upscaler = 'linear'; dlss = $false }
    'Balanced'    = @{ resolution = '1080p'; presentMode = 'fifo';    output120 = $false; precompile = $true; vramGeometry = $true; imagePoolMiB = 1024; textureCacheMiB = 4096; hostImportMiB = 12288; workingSetMiB = 0; upscaler = 'linear'; dlss = $false }
    'Quality'     = @{ resolution = '2160p'; presentMode = 'fifo';    output120 = $false; precompile = $true; vramGeometry = $true; imagePoolMiB = 4096; textureCacheMiB = 8192; hostImportMiB = 16384; workingSetMiB = 0; upscaler = 'linear'; dlss = $false }
}
$settings = $presets['Performance'].Clone()
if (Test-Path $path) {
    try { (Get-Content $path -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $settings[$_.Name] = $_.Value } } catch { }
}

$form = New-Object System.Windows.Forms.Form -Property @{ Text = 'Gran Turismo 7 - graphics settings'; StartPosition = 'CenterScreen'; FormBorderStyle = 'FixedDialog'; MaximizeBox = $false; AutoSize = $true; AutoSizeMode = 'GrowAndShrink'; Padding = '12,12,12,12'; Font = New-Object System.Drawing.Font('Segoe UI', 9) }
$grid = New-Object System.Windows.Forms.TableLayoutPanel -Property @{ ColumnCount = 2; AutoSize = $true; Dock = 'Fill' }
$tips = New-Object System.Windows.Forms.ToolTip -Property @{ AutoPopDelay = 20000 }
$row = 0
function Add-Row([string]$label, $control, [string]$tip) {
    $text = New-Object System.Windows.Forms.Label -Property @{ Text = $label; AutoSize = $true; Anchor = 'Left'; Margin = '0,6,12,6' }
    $control.Margin = '0,3,0,3'
    $grid.Controls.Add($text, 0, $script:row); $grid.Controls.Add($control, 1, $script:row); $script:row++
    if ($tip) { $tips.SetToolTip($text, $tip); $tips.SetToolTip($control, $tip) }
}
function New-Choice([string[]]$items, [string]$value) {
    $box = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 260 }
    $box.Items.AddRange($items); $box.SelectedItem = $value; if ($box.SelectedIndex -lt 0) { $box.SelectedIndex = 0 }
    $box
}
function New-Check([bool]$value) { New-Object System.Windows.Forms.CheckBox -Property @{ Checked = $value; AutoSize = $true } }
function New-Number([int]$value, [int]$max) { New-Object System.Windows.Forms.NumericUpDown -Property @{ Minimum = 0; Maximum = $max; Increment = 256; Value = [Math]::Min($value, $max); Width = 100 } }

$preset = New-Choice (@('Custom') + @($presets.Keys)) 'Custom'
Add-Row 'Preset' $preset 'Fills in every setting below.'
$resolution = New-Choice @('1080p', '2160p') $settings.resolution
Add-Row 'Rendering resolution' $resolution "The resolution GT7 renders at (the PS5's 1080p performance mode or its native 4K mode).`n1080p draws about a quarter of the pixels; the window scales the image to its size."
$upscaler = New-Choice @('linear', 'nearest') $settings.upscaler
Add-Row 'Upscaling filter' $upscaler "How the rendered image is scaled to the window. FSR 1 is planned; linear is the current default."
$dlss = New-Check $false; $dlss.Enabled = $false; $dlss.Text = 'not available yet'
Add-Row 'DLSS' $dlss "DLSS needs the game's per-pixel motion vectors and depth, which the port does not extract yet.`nIt stays off until it does."
$presentMode = New-Choice @('fifo', 'mailbox', 'immediate') $settings.presentMode
Add-Row 'Presentation (V-Sync)' $presentMode "fifo: V-Sync on, no tearing. mailbox: no tearing, lowest latency, can render past the refresh rate.`nimmediate: V-Sync off, may tear."
$output120 = New-Check ([bool]$settings.output120); $output120.Text = 'offer the game a 120 Hz display'
Add-Row '120 Hz output' $output120 'Lets GT7 enable its 120 Hz mode (experimental).'
$precompile = New-Check ([bool]$settings.precompile); $precompile.Text = 'build every cached shader before the title starts'
Add-Row 'Shader precompilation' $precompile "At boot, every shader and pipeline earlier sessions met is rebuilt for this GPU on all cores before the first frame,`ninstead of stalling a draw the first time it appears. Adds a minute or two to the boot."
$vramGeometry = New-Check ([bool]$settings.vramGeometry); $vramGeometry.Text = 'keep vertex and index data in VRAM'
Add-Row 'Geometry in VRAM' $vramGeometry 'Copies unchanged vertex and index ranges into video memory instead of reading them across PCIe on every draw.'
$imagePool = New-Number ([int]$settings.imagePoolMiB) 16384
Add-Row 'Texture pool (MiB)' $imagePool 'Video memory kept for reusing released images (render targets and textures) instead of reallocating them.'
$textureCache = New-Number ([int]$settings.textureCacheMiB) 16384
Add-Row 'Texture cache (MiB)' $textureCache "Sampled textures kept on the GPU (and their CPU snapshots) between uses. Too small, and textures are re-read and re-uploaded every frame:`n2 GiB thrashed in licence B-1; 6 GiB raised it from 1.24 to 1.55 FPS. Needs as much free RAM and VRAM."
$hostImport = New-Number ([int]$settings.hostImportMiB) 65536
Add-Row 'Shared guest memory (MiB)' $hostImport "How much of the game's memory the GPU may read in place (the PS5's unified memory). Lower it on GPUs with less VRAM."
$workingSet = New-Number ([int]$settings.workingSetMiB) 65536
Add-Row 'Process working set (MiB, 0 = auto)' $workingSet 'The minimum RAM kept resident for the game, so Windows does not page its memory out.'

$preset.Add_SelectedIndexChanged({
    $p = $presets[[string]$preset.SelectedItem]
    if (-not $p) { return }
    $resolution.SelectedItem = $p.resolution; $presentMode.SelectedItem = $p.presentMode; $upscaler.SelectedItem = $p.upscaler
    $output120.Checked = $p.output120; $precompile.Checked = $p.precompile; $vramGeometry.Checked = $p.vramGeometry
    $imagePool.Value = $p.imagePoolMiB; $textureCache.Value = $p.textureCacheMiB; $hostImport.Value = $p.hostImportMiB; $workingSet.Value = $p.workingSetMiB
})

$buttons = New-Object System.Windows.Forms.FlowLayoutPanel -Property @{ FlowDirection = 'RightToLeft'; AutoSize = $true; Dock = 'Fill'; Margin = '0,12,0,0' }
$launch = New-Object System.Windows.Forms.Button -Property @{ Text = 'Save and launch'; AutoSize = $true; DialogResult = 'Yes' }
$save = New-Object System.Windows.Forms.Button -Property @{ Text = 'Save'; AutoSize = $true; DialogResult = 'OK' }
$cancel = New-Object System.Windows.Forms.Button -Property @{ Text = 'Cancel'; AutoSize = $true; DialogResult = 'Cancel' }
if ($NoLaunch) { $launch.Visible = $false }
$buttons.Controls.AddRange(@($launch, $save, $cancel))
$grid.Controls.Add($buttons, 0, $row); $grid.SetColumnSpan($buttons, 2)
$form.Controls.Add($grid)
$form.AcceptButton = $launch; $form.CancelButton = $cancel

$result = $form.ShowDialog()
if ($result -eq 'Cancel') { return }
[ordered]@{
    resolution = [string]$resolution.SelectedItem; upscaler = [string]$upscaler.SelectedItem; dlss = $false
    presentMode = [string]$presentMode.SelectedItem; output120 = $output120.Checked; precompile = $precompile.Checked
    vramGeometry = $vramGeometry.Checked; imagePoolMiB = [int]$imagePool.Value; textureCacheMiB = [int]$textureCache.Value; hostImportMiB = [int]$hostImport.Value; workingSetMiB = [int]$workingSet.Value
} | ConvertTo-Json | Set-Content $path -Encoding utf8
"saved $path"
if ($result -eq 'Yes') { Start-Process powershell -ArgumentList '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'run-gt7.ps1') -WorkingDirectory $PSScriptRoot }

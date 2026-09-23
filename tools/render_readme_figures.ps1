param(
    [string]$RunDir = "runs/overnight_roster_visual_shaped_seed20260722",
    [string]$OutputDir = "docs/images",
    [int]$MaxUpdate = 34600
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$repo = Split-Path -Parent $PSScriptRoot
$runPath = [System.IO.Path]::GetFullPath((Join-Path $repo $RunDir))
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $repo $OutputDir))
# evaluations.csv is written by t8_v2_train at every evaluation; for runs that
# predate it, create it with: python tools\export_evaluations.py <run-dir>
$csvPath = Join-Path $runPath "evaluations.csv"
if (-not (Test-Path -LiteralPath $csvPath)) {
    throw "Missing $csvPath; backfill it with: python tools\export_evaluations.py $RunDir"
}
[System.IO.Directory]::CreateDirectory($outputPath) | Out-Null

$evaluations = @(
    Import-Csv -LiteralPath $csvPath | ForEach-Object {
        [pscustomobject]@{
            Update = [int]$_.update
            Deterministic = [double]$_.win_rate
            Stochastic = [double]$_.stochastic_win_rate
        }
    } | Where-Object { $_.Update -le $MaxUpdate }
)
if ($evaluations.Count -eq 0) {
    throw "No evaluation records found in $csvPath"
}

function New-Canvas {
    $bitmap = [System.Drawing.Bitmap]::new(1400, 800)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
    $graphics.Clear([System.Drawing.ColorTranslator]::FromHtml("#10161d"))
    return [pscustomobject]@{ Bitmap = $bitmap; Graphics = $graphics }
}

function New-Font([float]$size, [System.Drawing.FontStyle]$style = [System.Drawing.FontStyle]::Regular) {
    return [System.Drawing.Font]::new("Segoe UI", $size, $style, [System.Drawing.GraphicsUnit]::Pixel)
}

function New-Brush([string]$color) {
    return [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml($color))
}

function Save-Canvas($canvas, [string]$name) {
    $target = Join-Path $outputPath $name
    $canvas.Bitmap.Save($target, [System.Drawing.Imaging.ImageFormat]::Png)
    $canvas.Graphics.Dispose()
    $canvas.Bitmap.Dispose()
    Write-Output $target
}

$white = New-Brush "#f4f7fa"
$muted = New-Brush "#aeb9c5"
$gridPen = [System.Drawing.Pen]::new([System.Drawing.ColorTranslator]::FromHtml("#34414f"), 1)
$greenPen = [System.Drawing.Pen]::new([System.Drawing.ColorTranslator]::FromHtml("#66d296"), 5)
$pinkPen = [System.Drawing.Pen]::new([System.Drawing.ColorTranslator]::FromHtml("#ed7fb2"), 5)

$chart = New-Canvas
$g = $chart.Graphics
$g.DrawString("Held-out evaluation during overnight training", (New-Font 36 Bold), $white, 72, 44)
$g.DrawString(("{0} evaluations x 256 episodes | compatibility simulator | seed 20260722" -f $evaluations.Count), (New-Font 20), $muted, 75, 94)

$left = 110.0
$top = 155.0
$width = 1210.0
$height = 500.0
$minimumObserved = ($evaluations | ForEach-Object { $_.Deterministic; $_.Stochastic } | Measure-Object -Minimum).Minimum
$minimumRate = [Math]::Max(0.0, [Math]::Floor(($minimumObserved - 0.05) * 10.0) / 10.0)
$maximumRate = 1.0
$maximumUpdate = ($evaluations | Measure-Object Update -Maximum).Maximum

for ($rate = $minimumRate; $rate -le $maximumRate + 0.001; $rate += 0.1) {
    $y = $top + $height - (($rate - $minimumRate) / ($maximumRate - $minimumRate) * $height)
    $g.DrawLine($gridPen, $left, $y, $left + $width, $y)
    $g.DrawString(("{0:P0}" -f $rate), (New-Font 17), $muted, 44, $y - 12)
}
$tickInterval = [Math]::Max(100, [Math]::Ceiling(($maximumUpdate / 7.0) / 100.0) * 100)
for ($update = 0; $update -le $maximumUpdate; $update += $tickInterval) {
    $x = $left + (($update / $maximumUpdate) * $width)
    $g.DrawLine($gridPen, $x, $top, $x, $top + $height)
    $g.DrawString($update.ToString(), (New-Font 16), $muted, $x - 19, $top + $height + 14)
}

function Point-For($record, [string]$field) {
    $x = $left + (($record.Update / $maximumUpdate) * $width)
    $value = [double]$record.$field
    $y = $top + $height - (($value - $minimumRate) / ($maximumRate - $minimumRate) * $height)
    return [System.Drawing.PointF]::new([float]$x, [float]$y)
}

$deterministicPoints = [System.Drawing.PointF[]]@($evaluations | ForEach-Object { Point-For $_ "Deterministic" })
$stochasticPoints = [System.Drawing.PointF[]]@($evaluations | ForEach-Object { Point-For $_ "Stochastic" })
$g.DrawLines($greenPen, $deterministicPoints)
$g.DrawLines($pinkPen, $stochasticPoints)

$latest = $evaluations[-1]
$latestPoint = Point-For $latest "Deterministic"
$latestBrush = New-Brush "#66d296"
$g.FillEllipse($latestBrush, $latestPoint.X - 8, $latestPoint.Y - 8, 16, 16)
$g.DrawString(("Latest {0:P1}" -f $latest.Deterministic), (New-Font 19 Bold), $white, $latestPoint.X - 145, $latestPoint.Y - 46)
$g.DrawString("PPO update", (New-Font 18), $muted, 650, 707)
$g.DrawString("Deterministic", (New-Font 18 Bold), $latestBrush, 925, 727)
$pinkBrush = New-Brush "#ed7fb2"
$g.DrawString("Stochastic", (New-Font 18 Bold), $pinkBrush, 1090, 727)
$g.DrawString("Historical oscillation remains a release blocker despite the strong recent window.", (New-Font 17), $muted, 74, 752)
Save-Canvas $chart "v3-training-curve.png"

$benchmark = New-Canvas
$g = $benchmark.Graphics
$g.DrawString("RTX 5070 Ti isolated benchmark snapshot", (New-Font 38 Bold), $white, 72, 48)
$g.DrawString("Release build | median of 5 runs | trainer and visualizer stopped", (New-Font 20), $muted, 75, 101)

$cards = @(
    @{ X = 75; Y = 175; Label = "Simulator decisions/s"; Value = "429.2M"; Detail = "262,144 environments"; Color = "#66d296" },
    @{ X = 720; Y = 175; Label = "Simulated frames/s"; Value = "1.717B"; Detail = "4 frames per decision"; Color = "#ed7fb2" },
    @{ X = 75; Y = 430; Label = "README PPO decisions/s"; Value = "2.106M"; Detail = "4,096 envs, horizon 128"; Color = "#58a6ff" },
    @{ X = 720; Y = 430; Label = "Sustained PPO visits/s"; Value = "12.44M"; Detail = "32,768 envs, 20 updates"; Color = "#f4c95d" }
)
foreach ($card in $cards) {
    $rect = [System.Drawing.RectangleF]::new($card.X, $card.Y, 605, 205)
    $cardBrush = New-Brush "#18222d"
    $borderPen = [System.Drawing.Pen]::new([System.Drawing.ColorTranslator]::FromHtml("#34414f"), 2)
    $g.FillRectangle($cardBrush, $rect)
    $g.DrawRectangle($borderPen, $rect.X, $rect.Y, $rect.Width, $rect.Height)
    $accent = New-Brush $card.Color
    $g.FillRectangle($accent, $card.X, $card.Y, 10, 205)
    $g.DrawString($card.Label, (New-Font 21 Bold), $white, $card.X + 36, $card.Y + 27)
    $g.DrawString($card.Value, (New-Font 55 Bold), $accent, $card.X + 35, $card.Y + 67)
    $g.DrawString($card.Detail, (New-Font 18), $muted, $card.X + 38, $card.Y + 158)
    $cardBrush.Dispose()
    $borderPen.Dispose()
    $accent.Dispose()
}
$g.DrawString("Windows 11 | CUDA 13.4 driver API | NVIDIA 616.64 | Intel Core i5-12600K", (New-Font 18), $muted, 75, 704)
$g.DrawString("Sustained PPO: 205 W mean, 219 W peak, +1.61 GiB GPU memory.", (New-Font 17), $muted, 75, 745)
Save-Canvas $benchmark "v3-benchmarks.png"

$white.Dispose()
$muted.Dispose()
$gridPen.Dispose()
$greenPen.Dispose()
$pinkPen.Dispose()
$latestBrush.Dispose()
$pinkBrush.Dispose()

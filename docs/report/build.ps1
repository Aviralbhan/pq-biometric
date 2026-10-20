# Rebuilds the course report from eval/results/*.csv: figures -> .docx -> .pdf (PDF via Microsoft Word).
#   powershell -File docs/report/build.ps1 -NodeModules <folder that contains the "docx" npm package>
param([string]$NodeModules = "")
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path "$here\..\.."
& "$root\.venv\Scripts\python.exe" "$here\make_figures.py"
if ($NodeModules) { $env:NODE_PATH = $NodeModules }
node "$here\build_report.js"
if ($LASTEXITCODE -ne 0) { throw "build_report.js failed" }
$docx = "$here\PQ_Biometric_Course_Report.docx"
$pdf = "$here\PQ_Biometric_Course_Report.pdf"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $d = $word.Documents.Open($docx, $false, $true)
    "pages: $($d.ComputeStatistics(2))"
    $d.ExportAsFixedFormat($pdf, 17)
    $d.Close(0)
} finally { $word.Quit() }
"wrote $pdf"

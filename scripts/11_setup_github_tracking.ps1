param(
    [string]$Repo = "PaulHoang7/Brain_Segmentation"
)

$ErrorActionPreference = "Stop"

Write-Host "Setting up GitHub tracking labels and milestones for $Repo ..."

function Ensure-Label {
    param(
        [string]$Name,
        [string]$Color,
        [string]$Description
    )

    gh label create $Name --repo $Repo --color $Color --description $Description 2>$null
    if ($LASTEXITCODE -ne 0) {
        gh label edit $Name --repo $Repo --color $Color --description $Description
    }
}

$labels = @(
    @{ Name = "type:bug";        Color = "D73A4A"; Description = "Bug, defect, or regression" },
    @{ Name = "type:progress";   Color = "1D76DB"; Description = "Progress tracking update" },
    @{ Name = "type:task";       Color = "0E8A16"; Description = "Implementation task" },
    @{ Name = "type:experiment"; Color = "5319E7"; Description = "Ablation, benchmark, or experiment run" },
    @{ Name = "status:triage";   Color = "FBCA04"; Description = "New issue pending triage" },
    @{ Name = "status:in-progress"; Color = "0052CC"; Description = "Currently in progress" },
    @{ Name = "status:blocked";  Color = "B60205"; Description = "Blocked by dependency or resource" },
    @{ Name = "status:review";   Color = "C5DEF5"; Description = "Ready for review" },
    @{ Name = "status:done";     Color = "0E8A16"; Description = "Completed and validated" },
    @{ Name = "priority:p0";     Color = "B60205"; Description = "Critical priority" },
    @{ Name = "priority:p1";     Color = "D93F0B"; Description = "High priority" },
    @{ Name = "priority:p2";     Color = "FBCA04"; Description = "Normal priority" }
)

foreach ($label in $labels) {
    Ensure-Label -Name $label.Name -Color $label.Color -Description $label.Description
}

$milestones = @(
    "W01-W02 preprocess",
    "W03-W04 vanilla-sam",
    "W05-W06 prompt-generator",
    "W07-W10 sam-lora-cascade",
    "W11-W12 nnunet-baseline",
    "W13-W14 ablation-robustness",
    "W15-W16 final-results-writeup"
)

$existingMilestones = gh api "repos/$Repo/milestones?state=all&per_page=100" | ConvertFrom-Json
$existingTitles = @($existingMilestones | ForEach-Object { $_.title })

foreach ($title in $milestones) {
    if ($existingTitles -contains $title) {
        Write-Host "Milestone already exists: $title"
    } else {
        gh api "repos/$Repo/milestones" --method POST -f title="$title" | Out-Null
        Write-Host "Created milestone: $title"
    }
}

Write-Host "Done. You can now track progress and bugs in GitHub Issues."

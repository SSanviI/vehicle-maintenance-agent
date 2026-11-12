$dirs = @(
    'backend/app/api',
    'backend/app/agent',
    'backend/app/llm/providers',
    'backend/app/ml',
    'backend/app/context',
    'backend/app/memory',
    'backend/app/database',
    'backend/app/schemas',
    'backend/app/services',
    'backend/app/core',
    'backend/tests',
    'frontend',
    'data',
    'models',
    'ml/notebooks',
    'ml/reports',
    'ml/artifacts',
    'scripts',
    'docs',
    'tests'
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}
Write-Host 'All directories created.'

Copy-Item 'd:\Users\vehicle_maintenance_platform\data\vehicle_maintenance_data.csv' -Destination 'data\vehicle_maintenance_data.csv' -Force
Write-Host 'Dataset copied.'

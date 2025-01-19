*** Settings ***
Library    SeleniumLibrary
Library    DateTime

*** Variables ***
${SCREENSHOT_DIR}  data/screenshots/
${BROWSER}         Chrome

*** Test Cases ***
Capture Screenshots
    [Documentation]    Capture screenshots at various steps of the UI testing
    [Tags]    Screenshot
    Open Browser    https://example.com    ${BROWSER}
    Capture Screenshot With Timestamp
    # Add more steps as needed, each can capture a screenshot
    Capture Screenshot With Timestamp
    Close Browser

*** Keywords ***
Capture Screenshot With Timestamp
    [Documentation]    Capture a screenshot and save it with a timestamped filename
    ${timestamp}=    Get Current Date    result_format=%Y%m%d_%H%M%S
    ${screenshot}=    Set Variable    screenshot_${timestamp}.png
    Capture Page Screenshot    ${SCREENSHOT_DIR}${screenshot}
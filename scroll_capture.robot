*** Settings ***
Library    SeleniumLibrary

*** Variables ***
${URL}    http://example.com
${ZOOM_LEVEL}    1.5  # 1.0 is 100%, 1.5 is 150%, etc.

*** Test Cases ***
Zoom Page And Center Scrollbar
    Open Browser    ${URL}    Chrome
    Maximize Browser Window
    Zoom Page    ${ZOOM_LEVEL}
    Center Horizontal Scrollbar
    # Optional: Take a screenshot or perform other actions
    Close Browser

*** Keywords ***
Zoom Page
    [Arguments]    ${zoom_level}
    Execute JavaScript    document.body.style.zoom='${zoom_level}';

Center Horizontal Scrollbar
    # Calculate the scroll position based on the zoom level and page width
    ${scroll_position}=    Calculate Scroll Position
    Execute JavaScript    window.scrollTo(${scroll_position}, 0)

Calculate Scroll Position
    ${viewport_width}=    Get Viewport Width
    ${document_width}=    Get Document Width
    ${scroll_position}=    (${document_width} - ${viewport_width}) / 2
    [Return]    ${scroll_position}

Get Viewport Width
    ${viewport_width}=    Execute JavaScript    return window.innerWidth;
    [Return]    ${viewport_width}

Get Document Width
    ${document_width}=    Execute JavaScript    return document.body.scrollWidth;
    [Return]    ${document_width}
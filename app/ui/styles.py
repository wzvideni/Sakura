from __future__ import annotations


PET_WINDOW_STYLEHEET = """
#speechBubble {
    background: rgba(255, 232, 241, 220);
    border: 1px solid rgba(238, 172, 200, 158);
    border-radius: 26px;
}
#speakerName {
    color: #d55b91;
    font-size: 13px;
    font-weight: 700;
}
#speechText {
    color: #4b3440;
    font-size: 19px;
    line-height: 1.35;
}
#inputBar {
    background: transparent;
    border: none;
}
#petInput {
    background: rgba(255, 255, 255, 96);
    border: 1px solid rgba(255, 255, 255, 218);
    border-radius: 19px;
    color: #2f2630;
    font-size: 15px;
    font-weight: 700;
    padding: 3px 16px;
    selection-background-color: rgba(74, 170, 214, 185);
}
#petInput:focus {
    background: rgba(255, 255, 255, 132);
    border: 1px solid rgba(74, 170, 214, 230);
}
#petInput:disabled {
    color: rgba(47, 38, 48, 150);
}
#sendButton {
    background: rgba(74, 170, 214, 225);
    border: none;
    border-radius: 16px;
    color: white;
    font-size: 15px;
    font-weight: 800;
    min-width: 68px;
    padding: 4px 14px;
}
#sendButton:hover {
    background: rgba(48, 145, 195, 235);
}
#sendButton:disabled {
    background: rgba(126, 171, 193, 190);
}
#screenshotButton {
    background: rgba(255, 255, 255, 116);
    border: 1px solid rgba(255, 255, 255, 218);
    border-radius: 16px;
    color: #4b3440;
    font-size: 15px;
    font-weight: 800;
    min-width: 58px;
    padding: 4px 12px;
}
#screenshotButton:hover {
    background: rgba(255, 255, 255, 150);
}
#screenshotButton[screenshotAttached="true"] {
    background: rgba(93, 181, 130, 225);
    border: none;
    color: white;
}
#screenshotButton:disabled {
    background: rgba(176, 181, 184, 150);
    color: rgba(75, 52, 64, 135);
}
#confirmActionButton {
    background: rgba(93, 181, 130, 225);
    border: none;
    border-radius: 16px;
    color: white;
    font-size: 15px;
    font-weight: 800;
    min-width: 58px;
    padding: 4px 12px;
}
#cancelActionButton {
    background: rgba(180, 130, 146, 210);
    border: none;
    border-radius: 16px;
    color: white;
    font-size: 15px;
    font-weight: 800;
    min-width: 58px;
    padding: 4px 12px;
}
"""

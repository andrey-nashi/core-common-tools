//---- Launch 'core.js' when opening a specific URL
chrome.webNavigation.onCompleted.addListener(function(details) {
    chrome.tabs.executeScript(details.tabId, {
	file: 'core.js'
    });
}, {
    url: [{
        hostContains: '.example'
    }],
});


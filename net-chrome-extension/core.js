//---------------------------------------------------------------------
// ---- Useful functions for data scrapping, contend downloading
//---------------------------------------------------------------------

function sleep(milliseconds) {
  const date = Date.now();
  let currentDate = null;
  do {
    currentDate = Date.now();
  } while (currentDate - date < milliseconds);
}

function download(content, file_name) {
    //---- Download content, save into file
    if (content != "") {
    let a = document.createElement('a');
    a.href = "data:application/octet-stream,"+encodeURIComponent(content);
    a.download = file_name;
    a.click();
    }
}

function redirect(url, time) {
    //---- Redirect with timeout
    sleep(time)
    window.location.replace(url);
}

//---------------------------------------------------------------------

//Examples on how to grab elements from web page
//var element = document.getElementsByClassName("table-responsive");
//var content = element[0].innerText;
//var rows = document.getElementsByTagName("table")[0].rows;

//---------------------------------------------------------------------
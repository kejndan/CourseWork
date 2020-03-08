function loadXMLDoc()
{
var xmlhttp;
if (window.XMLHttpRequest)
  {// код для IE7+, Firefox, Chrome, Opera, Safari
  xmlhttp=new XMLHttpRequest();
  }
else
  {// код для IE6, IE5
  xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
  }
xmlhttp.onreadystatechange=function()
  {
  if (xmlhttp.readyState==4 && xmlhttp.status==200)
    {
      // var enc = new string_transcoder("windows-1252");
      // var tenc = ;
    document.getElementById("window_output").innerHTML=xmlhttp.responseText;
    }
  }
xmlhttp.open("GET","http:\\media\\output.txt",true); // true - используем АСИНХРОННУЮ передачу
xmlhttp.send();
}

function timer() {
	// var elem = document.getElementById('window_output');
    //
	// elem.value = parseInt(elem.value)+1; //parseInt преобразует строку в число
    loadXMLDoc()

};




// var reader = new FileReader();
// reader.onload = function(event) {
//     var contents = event.target.result;
//     console.log("Содержимое файла: " + contents);
// };
//
// reader.onerror = function(event) {
//     console.error("Файл не может быть прочитан! код " + event.target.error.code);
// };
setInterval(timer, 5000)
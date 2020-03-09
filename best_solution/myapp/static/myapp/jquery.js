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
      var arr = xmlhttp.responseText.split(/\r?\n/);
      console.log(arr);
      for (let i=window.stopIndex; i<arr.length; i++){
          document.getElementById("window_output").innerHTML += '<p>'+arr[i]+'</p>';
      }
      window.stopIndex=arr.length-1;
    }
  }
xmlhttp.open("GET","http:\\media\\output.txt",true); // true - используем АСИНХРОННУЮ передачу
xmlhttp.send();
}

function timer() {
    loadXMLDoc()
};
window.stopIndex = 0;
setInterval(timer, 2000)
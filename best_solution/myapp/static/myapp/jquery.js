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
// // function startUpdateOutput()
// // {
// //   document.getElementById("start_btn").clicked = true;
// //   console.log('sadasdas');
// //
// // }
// // if (document.getElementById("start_btn").clicked === true) {
// //   setInterval(timer, 2000);
// // }
// //
// alert(src);
// $("#start_btn").on('click', function(event) {
//   // Отменяем отправку по нажатию.
//   event.preventDefault();
//
//   // Тут выполняем нужные действия:
//   // инициируем нажатие на объект #savers.
//   $("#start_btn").trigger('click');
//
//   // Отправляем форму.
//   // $(this).closest('form').submit();
// });
// // // document.getElementById('bot').onclick =
$('#form').submit(function(e){
//     e.preventDefault();
//     $.post('/processing/', $(this).serialize(), function (data) {
//        document.getElementById('start_btn').value = 'RUNNING';
//     });
// var datastring = $form.serialize();
    e.preventDefault();
    alert('sad')
    $.ajax({  //Call ajax function sending the option loaded
      url: "/processing/",  //This is the url of the ajax view where you make the search
      type: 'POST',
      data: datastring,
        success: function() {
            }
        })
    });
console.log(window.location.pathname);
if (window.location.pathname === '/processing/working') {
    document.addEventListener("DOMContentLoaded", function(event) {
    document.getElementById("stop_btn").disabled = 0;
    // document.getElementById("start_btn").disabled = 1;
    });
    setInterval(timer, 2000);

}


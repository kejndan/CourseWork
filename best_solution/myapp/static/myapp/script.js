function loadXMLDoc()
{
    var xmlhttp;
    if (window.XMLHttpRequest) {// код для IE7+, Firefox, Chrome, Opera, Safari
    xmlhttp=new XMLHttpRequest();
    }
    else {// код для IE6, IE5
    xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
    }
    var max_elem = 90;
    xmlhttp.onreadystatechange=function() {
    if (xmlhttp.readyState==4 && xmlhttp.status==200) {
        var arr = xmlhttp.responseText.split(/\r?\n/);

        if (arr[arr.length-1] === '##END##') {
        window.location.href = 'result';
        }
        else {
            console.log(arr.length-window.stopIndex);
            var elem = document.getElementById("myBar");
            window.width_bar += (arr.length-window.stopIndex)/max_elem*100;
            elem.style.width = window.width_bar + "%";
            console.log(elem);
            console.log(arr);
        }

      window.stopIndex=arr.length;
    }
  }
xmlhttp.open("GET","http:\\media\\output.txt",true); // true - используем АСИНХРОННУЮ передачу
xmlhttp.send();
}

function timer() {
    loadXMLDoc()
};
window.stopIndex = 0;
window.width_bar = 0;
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
    $.ajax({  //Call ajax function sending the option loaded
      url: "/processing/",  //This is the url of the ajax view where you make the search
      type: 'POST',
      data: datastring,
        success: function() {
            }
        })
    });
if (window.location.pathname === '/processing/') {
    document.addEventListener("DOMContentLoaded", function(event) {
    // document.getElementById("start_btn").disabled = 1;
    });
    setInterval(timer, 2000);
}


var expanded_check = false;
function showCheckboxes() {
  var checkboxes = document.getElementById("checkboxes");
  if (!expanded_check) {
        checkboxes.style.display = "block";
        expanded_check = true;
        closeAnother("checkboxes")

  } else {
    checkboxes.style.display = "none";
    expanded_check = false;
  }
}

var expanded_radio = false;
function showRadiobuttons() {
  var checkboxes = document.getElementById("radiobuttons");
  if (!expanded_radio) {
    checkboxes.style.display = "block";
    expanded_radio = true;
   closeAnother("radiobuttons");
  } else {
    checkboxes.style.display = "none";
    expanded_radio = false;
  }
}
var expanded_checkboxes_for_preprocessing = false;
function showCheckboxesPreprocessing() {
  var checkboxes = document.getElementById("checkboxes_for_preprocessing");
  if ($('#on_processing_missing').is(':checked')) {
      if (!expanded_checkboxes_for_preprocessing) {
          checkboxes.style.display = "block";
          expanded_checkboxes_for_preprocessing = true;
          closeAnother("checkboxes_for_preprocessing");
      } else {
          checkboxes.style.display = "none";
          expanded_checkboxes_for_preprocessing = false;
      }
  }
}

var expanded_checkboxes_for_handling_outliners = false;
function showCheckboxesHandlingOutliners() {
  var checkboxes = document.getElementById("checkboxes_for_handling_outliners");
  if ($('#on_handling_outliners').is(':checked')) {
      if (!expanded_checkboxes_for_handling_outliners) {
          checkboxes.style.display = "block";
          expanded_checkboxes_for_handling_outliners = true;
          closeAnother("checkboxes_for_handling_outliners");
      } else {
          checkboxes.style.display = "none";
          expanded_checkboxes_for_handling_outliners = false;
      }
  }
}

var expanded_checkboxes_for_binning = false;
function showCheckboxesBinning() {
  var checkboxes = document.getElementById("checkboxes_for_binning");
  if ($('#on_binning').is(':checked')) {
      if (!expanded_checkboxes_for_binning) {
          checkboxes.style.display = "block";
          expanded_checkboxes_for_binning = true;
          closeAnother("checkboxes_for_binning");
      } else {
          checkboxes.style.display = "none";
          expanded_checkboxes_for_binning = false;
      }
  }
}

var expanded_checkboxes_for_transform = false;
function showCheckboxesTransform() {
  var checkboxes = document.getElementById("checkboxes_for_transform");
  if ($('#on_transform').is(':checked')) {
      if (!expanded_checkboxes_for_transform) {
          checkboxes.style.display = "block";
          expanded_checkboxes_for_transform = true;
          closeAnother("checkboxes_for_transform");
      } else {
          checkboxes.style.display = "none";
          expanded_checkboxes_for_transform = false;
      }
  }
}


var expanded_checkboxes_for_scaling = false;
function showCheckboxesScaling() {
  var checkboxes = document.getElementById("checkboxes_for_scaling");
  if ($('#on_scaling').is(':checked')) {
      if (!expanded_checkboxes_for_scaling) {
          checkboxes.style.display = "block";
          expanded_checkboxes_for_scaling = true;
          closeAnother("checkboxes_for_scaling");
      } else {
          checkboxes.style.display = "none";
          expanded_checkboxes_for_scaling = false;
      }
  }
}

function closeAnother(name){
    if (name !=="checkboxes"){
        document.getElementById("checkboxes").style.display = "none";
        expanded_check = false;
    }
    if (name !== "radiobuttons"){
        document.getElementById("radiobuttons").style.display = "none";
        expanded_radio = false;
    }
    if (name !== "checkboxes_for_preprocessing"){
        document.getElementById("checkboxes_for_preprocessing").style.display = "none";
        expanded_checkboxes_for_preprocessing = false;
    }
    if (name !== "checkboxes_for_handling_outliners"){
        document.getElementById("checkboxes_for_handling_outliners").style.display = "none";
        expanded_checkboxes_for_handling_outliners = false;
    }
    if (name !== "checkboxes_for_binning"){
        document.getElementById("checkboxes_for_binning").style.display = "none";
        expanded_checkboxes_for_binning = false;
    }
    if (name !=="checkboxes_for_transform"){
        document.getElementById("checkboxes_for_transform").style.display = "none";
        expanded_checkboxes_for_transform = false;
    }
    if (name !=="checkboxes_for_scaling"){
        document.getElementById("checkboxes_for_scaling").style.display = "none";
        expanded_checkboxes_for_scaling = false;
    }
}

$(function() {
$("#check_all_features").change(function(){
    $('.checkbox_features').prop('checked', $(this).is(':checked'));
});
});

$(function() {
$("#check_all_preprocessing").change(function(){
    $('.checkbox_preprocessing').prop('checked', $(this).is(':checked'));
});
});

$(function() {
$("#check_all_handling_outliners").change(function(){
    $('.checkbox_handling_outliners').prop('checked', $(this).is(':checked'));
});
});

$(function() {
$("#check_all_binning").change(function(){
    $('.checkbox_binning').prop('checked', $(this).is(':checked'));
});
});

$(function() {
$("#check_all_transform").change(function(){
    $('.checkbox_transform').prop('checked', $(this).is(':checked'));
});
});

$(function() {
$("#check_all_scaling").change(function(){
    $('.checkbox_scaling').prop('checked', $(this).is(':checked'));
});
});
$(function(){

$("div[class='upload_block']").click(function () {
    $('input[name="file"]').trigger('click');
});
});


$(function(){

$("div[id='Pipeline 1']").click(function () {
    $('input[name="Pipeline 1"]').trigger('click');
});
});

$(function(){

$("div[id='Pipeline 2']").click(function () {
    $('input[name="Pipeline 2"]').trigger('click');
});
});

$(function(){

$("div[id='Pipeline 3']").click(function () {
    $('input[name="Pipeline 3"]').trigger('click');
});
});
function loadXMLDoc()
{
    var xmlhttp;
    if (window.XMLHttpRequest) {// код для IE7+, Firefox, Chrome, Opera, Safari
    xmlhttp=new XMLHttpRequest();
    }
    else {// код для IE6, IE5
    xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
    }
    xmlhttp.onreadystatechange=function() {
    if (xmlhttp.readyState==4 && xmlhttp.status==200) {
        var arr = xmlhttp.responseText.split(/\r?\n/);
        for (let i=window.stopIndex; i<arr.length; i++){
            if (arr[i] === '##END##') {
            window.location.href = 'result';
            }
            else {
            document.getElementById("window_output").innerHTML += '<p>'+arr[i]+'</p>';}
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
    $.ajax({  //Call ajax function sending the option loaded
      url: "/processing/",  //This is the url of the ajax view where you make the search
      type: 'POST',
      data: datastring,
        success: function() {
            }
        })
    });
if (window.location.pathname === '/processing/working') {
    document.addEventListener("DOMContentLoaded", function(event) {
    document.getElementById("stop_btn").disabled = 0;
    // document.getElementById("start_btn").disabled = 1;
    });
    setInterval(timer, 2000);
}


var expanded_check = false;
function showCheckboxes() {
  var checkboxes = document.getElementById("checkboxes");
  if (!expanded_check) {
      console.log('block');
    checkboxes.style.display = "block";
    expanded_check = true;
  } else {
      console.log('none');
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
      } else {
          checkboxes.style.display = "none";
          expanded_checkboxes_for_scaling = false;
      }
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
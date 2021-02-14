$(document).ready(function () {
  // back to top button
  $(window).scroll(function () {
    if ($(this).scrollTop() > 100) {
      $("#scroll").fadeIn();
    } else {
      $("#scroll").fadeOut();
    }
  });
  $("#scroll").click(function () {
    $("html, body").animate({ scrollTop: 0 }, 600);
    return false;
  });

  // logic for collapsable Headers
  $(".col_head").click(function () {
    $(this).nextAll("div .col_con").first().slideToggle("fast");
  });
});

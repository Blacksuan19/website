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
    var current = $(this).nextAll("div .col_con").first();
    $(this)
      .parent()
      .find(".col_con")
      .each(function () {
        if (current.is(this)) {
          $(this).slideToggle("fast");
        } else {
          $(this).slideUp("fast");
        }
      });
  });

  // contact form validation
  $(document).ready(function () {
    $("#submit-button").click(function () {
      $("#contact-form").validate({
        rules: {
          name: {
            required: true,
          },
          email: {
            required: true,
            email: true,
          },
          message: {
            required: true,
          },
        },
        submitHandler: function () {
          return true;
        },
      });
    });
  });
});

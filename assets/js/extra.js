$(document).ready(function () {
  $("[data-share]").each(function () {
    var shareRoot = this;
    var shareButton = shareRoot.querySelector("[data-share-trigger]");
    var shareTitle = shareRoot.getAttribute("data-share-title") || document.title;
    var shareUrl = shareRoot.getAttribute("data-share-url") || window.location.href;
    var canUseNativeShare = typeof navigator.share === "function";

    if (!shareButton) {
      return;
    }

    if (!canUseNativeShare) {
      shareRoot.setAttribute("hidden", "hidden");
      return;
    }

    shareButton.removeAttribute("hidden");

    shareButton.addEventListener("click", function () {
      navigator.share({
        title: shareTitle,
        url: shareUrl,
      }).catch(function (error) {
        if (error && error.name === "AbortError") {
          return;
        }

        shareRoot.setAttribute("hidden", "hidden");
      });
    });
  });

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

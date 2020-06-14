---
layout: page
title: Contact
nav-menu: true
---

<!-- Contact -->
<section id="contact">
  <div class="inner">
    <section>
    {% if site.home-about == false %}
                <header style="margin-top: -40px" class="major">
                    <h2>About</h2>
                </header>
                <p>{{ site.about-text }}</p>
    {% endif %}
    
    For any Questions, enquiries, Contact me via the form below or send me a direct email. you can also get in touch with me via telegram.<br/><br/>
      <form action="https://formspree.io/{{ site.email }}" method="POST">
        <div class="field half first">
          <label for="name">Name</label>
          <input type="text" name="name" id="name" />
        </div>
        <div class="field half">
          <label for="email">Email</label>
          <input type="text" name="_replyto" id="email" />
        </div>
        <div class="field">
          <label for="message">Message</label>
          <textarea name="message" id="message" rows="6"></textarea>
        </div>
        <ul class="actions">
          <li><input type="submit" value="Send Message" class="special" /></li>
          <li><input type="reset" value="Clear" /></li>
        </ul>
      </form>
    </section>
    <section class="split">
      <section>
        <div class="contact-method">
          <span class="icon alt fa-envelope"></span>
          <h3>Email</h3>
          <a href="mailto:{{ site.email }}">{{ site.email }}</a>
        </div>
      </section>
      <section>
        <div class="contact-method">
          <span class="icon alt fa-telegram"></span>
          <h3>Telegram</h3>
          <span><a href="{{ site.telegram_url }}">{{ site.telegram_url | split: '/' | last | capitalize }}</a></span>
        </div>
      </section>
      <section>
        <div class="contact-method">
          <span class="icon alt fa-home"></span>
          <h3>Address</h3>
          <span>
            {% if site.street_address %}
            {{ site.street_address }}<br />
            {% endif %} {% if site.city %}
            {{ site.city }}, {% endif %} {% if site.state %}
            {{ site.state }}
            {% endif %} {% if site.zip_code %}
            {{ site.zip_code }}<br />
            {% endif %} {% if site.country %}
            {{ site.country }}
            {% endif %}
          </span>
        </div>
      </section>
    </section>

  </div>
</section>

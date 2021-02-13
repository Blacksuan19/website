---
layout: post
title: How to Set up a Telegram Userbot on heroku
description: Extending your functionality
image: /assets/images/heroku.png
project: false
permalink: /blog/:title/
tags:
  - telegram
  - userbot
  - heroku
  - python
---

Heroku is a container-based cloud platform for deploying, managing and scaling
modern apps. <br> A telegram user bot is not actually a bot (it can be), it uses
commands to communicate with the user, for example you can do `.weather city`
and it will send you the weather report for that location, it can convert
currencies, translate text and do a lot more. <br> The setup process isn't that
complicated but its not widely shared, you have to give it a go on your own,
anyway i am about to change that here. <br> There are many forks of the userbot,
but i personally use [this one](http://GitHub.com/Nick80835/Telegram-UserBot)
because it doesn't use a database, which i don't need, however without a
database some functions won't work.

### How To setup the bot:

- First of course we need to clone the repo `git clone http://GitHub.com/Nick80835/Telegram-UserBot`
- then install the required packages `pip3 install -r requirements.txt`
- next you need to add your username and other stuff to the config file
- first before editing rename the file `mv sample_config.env config.env`
- how to obtain all the codes is explained in the env file
- getting telegram API_KEY and API_HASH
  - login to the [my.telegram.org](https://my.telegram.org)
  - select API development tools
  - create an app and fill out the details (you can put anything in url)
  - copy you codes to config.env
- now you can start the bot with `python3 -m userbot`
- congrats! you have a bot but its hosted on your pc which you don't run all day (if you do then stop reading)

The bot uses the userbot.session file so you wont have to login again, add that
file to the repo first (do not push this file to github!).<br>

### How to deploy to heroku:

- create and account
- create an app with any name
- while selecting the git choose heroku git
- download the [cli tool](https://devcenter.heroku.com/articles/heroku-cli)
- login to cli tool `heroku login`
- navigate to the bot folder you cloned earlier
- switch to master branch or create it if it doesn't exist (heroku only runs apps from master branch) `git branch -m master`
- push the source to your heroku app `git push heroku master`(force push if you got any errors `git push -f heroku master`)
- now your bot source and session file are on heroku git you can start it `heroku run python3 -m userbot`
- your bot should run fine now!
- if the bot doesnt start, maybe the environment isn't getting set up so add your config to heroku env variables `heroku config:set CONFIG=value`
- to run the bot in the heroku background use `heroku run:detached python3 -m userbot`
- congrats! your bot is now running on heroku.

#### Note:

app running on heroku is not a git repository so you cant use the .update
command, to update the bot first stop the running bot process `heroku stop processid` (to get the id run `heroku ps` run.XXXX is the id), and then you can
update normally with `git pull`. <br> <br> and thats all your bot should be
kicking fine for now, heroku has a limited amount of hours for free accounts
(550 hours), it will also kill all processes if there is no activity for more
than 30min or so, don't worry you can just restart it again

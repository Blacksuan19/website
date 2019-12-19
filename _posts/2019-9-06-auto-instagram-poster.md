---
layout: post
title: Auto Instagram Poster
description: Automating the Boring Stuff with Python
image: /assets/images/instagram.png
project: false
permalink: /blog/:title/
---
So few days ago i had to market my new instagram account, i needed to post every
hour or so. and instagram doesn't exactly help with bulk posting especially from
PC, and it was frustrating to do from mobile and instagram never suppoted
uploading from desktop. so i got fed up and decided to write a simple script
that will take a photo, caption and a timer.<br> <br> The script will sleep for
the amount of specified time and after that post the photos,(yes i could've done
this on an async way but ain't nobody got time for that), anyway i had to find a
way to upload phots from pc and as a linux user i just googled instagram cli
(works all the time) and boom here is
[instapy](https://github.com/instagrambot/instapy-cli) a cli tool that uploads
photos or videos with a caption.<br> <br> Exactly what the doctor ordered, now
it was time to write a wrapper that will accept those argument's and foreword
them to instapy. at first i thought bash will do the trick since i am a big bash
fan (checkout my [scripts](https://github.com/Blacksuan19/Scripts) btw), but
there was this weird issue with captions where only the first word will be
posted (issue already opened).<br><br> my only option now was python, i am not a
fan of python (i am the java guy) but modern problems require modern solutions,
i wrote the script in python and it was easy enough that i wrote some help
documentation.<br>

#### Previous Way
- find the product
- download photos
- if photos not high quality go back to supplier images
- send phots to my self via telegram
- copy and send description via telegram
- open my phone and download the photos and copy caption
- open up [preview](https://play.google.com/store/apps/details?id=com.sensio.instapreview&hl=en) and set up the posts with captions hashtags etc
- wait for the time notification and then manually post

#### The thug life

With this i just provide a photo a .txt with the caption and a number to post
the photo after (60 min, 120 min etc), and now i can chill out and watch more
random psychology stuff on Youtube.<br> the only caveat is you can only upload
one photo per post but that should do fine for now.<br>
<pre>
<code class="language-python">
import sys
import time
from instapy_cli import client
# check if there are no args
if len(sys.argv) < 2:
    print("Empty Arguments!")
    print("Please select a pic, caption file and delay")
    print("example usage python3 upload.py photo.png caption.txt 1")
    print("delay has to be > 0")
    print("tip: attach the process to the terminal or disown it when you have a timer (it will upload after whatever time you set)")
    sys.exit()
username = 'USERNAME'
password = 'PASSWORD'
image = sys.argv[1]
captionfile = sys.argv[2]
delay = sys.argv[3]
with open(captionfile, 'r') as file:
    caption = file.read()
with client(username, password) as cli:
    time.sleep(int(delay) * 60)
    cli.upload(image, caption)
</code>
</pre>
<br>
this is an example of making laziness work on your favour, i still cant imagine
myself posting those images one by one like bruh.<br> and again
> modern problems require modern solutions

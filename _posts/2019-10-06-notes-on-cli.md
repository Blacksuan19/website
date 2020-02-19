---
layout: post
title: Note Taking on the CLI
description: how to minimal note taking setup
image: /assets/images/sncli.png
project: false
permalink: /blog/:title/
tags: notes cli simplenote linux
---

So the semester just started and i had to ditch my old note taking setup because
it didn't have any possible way of a 2-way sync and no mobile client, i had many
options to choose from but in the end i narrowed it down to just a few, my
criteria for a note taking setup was:

- have a cli app (i am moving from GUI)
- markdown support (i just can't go back to ever note)
- full sync hosted by them or a simple way i could host it myself
- a good android client

There are few apps which fit this description including, [inkdrop notes](https://inkdrop.app/),
[simplenotes](https://simplenote.com/),  [notable](https://github.com/notable/notable) and [notion](https://www.notion.so/), i have tried most of
these and these apps fit the
basic criteria but most of them lacked my first requirement which is a cli app
the only app which fit that was simple notes with something called sncli

##### why not qualified?
- inkdrop: no cli app, not entirely free and the desktop app is super heavy
    (damn you electron)
- notable: no cli app, and no sync or android app.
- notion: no cli app, again electron so heavier than the new planet 9 black hole,
    although its very feature rich and i might use it if i had a better
    machine.

as you can see most of these apps have their strengths and flaws, but for me
simple notes was the obvious choice even before i knew there is a cli client for
it and that's for few reasons:
- before switching to evernote i was using simple note (that was like 3-4 years
    ago)
- their sync model is simple(get it) and works on all platforms (you could also
    make your own client easily with their api)
- simple interface that doesn't get in the way (i am looking at you notable)

##### why not host on Dropbox or anywhere else?
Well, that's a good question, but i wanted something native, something that has
sync built in without having to go and do the actual implementation also i don't use Dropbox
myself and i am not willing to switch to it just for notes, if i wanted to use
Dropbox i would've just used Typora at this point.

#### Sncli
This is it, the moment i saw the git repo of this i knew its gonna be good, and
it is, it keeps everything synced at all times, the changes are reflected at an
instant, fully written in python with a simple tui that you don't actually edit
on, this is everything i asked for, however it's not all rainbows and shine, the
issues include but not limited to:

- failure to sync at some occasions
- sometimes newly created notes won't be uploaded and will be discarded so you
    have to type it again (that's the stupidest bug i have ever seen)
- the app hangs while its syncing the notes(the GUI and the sync backend are
    running on the same thread)
- takes a while to start


Editing notes work by copying the note to the temp folder and then opening it in
your specified editor.
The app itself doesn't have a notes editor it can just view a note, change tags
rename an do other things, for editing you need to specify an editor, this is
where nvim comes into the picture, i have configs that will set the file type
and other stuff for all files that have the word sncli and are stored at tmp
currently.

The key bindings follow vim style so its easy to use and get used to, you can
edit all the key bindings in the sncli config file.

<img src="/assets/images/bindings.png" alt="keyboard shortcuts">

#### building my own simplenote client
As i mentioned earlier simplenote is great, their api is great and some guy made
a [python API](https://github.com/mrtazz/simplenote.py) for their sync web services which makes it even better.
i got my hands dirty by trying to do a simple cli client implementation, which
didn't get much anywhere (python isn't exactly my favorite's language)

```python
import simplenote
import json
import re
sn = simplenote.Simplenote('email', 'password')
notes_id = sn.get_note_list(data=False)
first = str(notes_id[0])

# some next level regex to split the string to a list of key value pairs (pls dont touch ma spaghet)
regex = r"\{(.*?)\}"
matches = re.finditer(regex, first, re.MULTILINE | re.DOTALL)
matches_list = []
for matchNum, match in enumerate(matches):
    for groupNum in range(0, len(match.groups())):
        matches_list.append(match.group(1).replace("'key':", "").replace("'version':", "").replace("'", ""))

ids = []
for pair in matches_list:
    ids.append(pair.split(",")[0].strip())
print(ids[4])

current = sn.get_note(ids[1])
print(current)
```
<br>
this gives me all the current notes id's with which i could get the note
content, tags, history and more, the API is very flexible and can do great
things.
The plan is to have the backend on a separate thread and the GUI on another so
the app won't hang while syncing, tho i still don't know exactly how python
threads work, however i am going to be working on this for a while.
##### conclusion
in the end i kinda found what i am looking for, sncli will do for now but i am
still on the search for something that's more stable while at the same time
working on my own cli client which if actually becomes something big i might
release otherwise lets pretend it didn't happened.

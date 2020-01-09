---
layout: post
title: Going Back To Plasma
description: But with a twist
image: /assets/images/kde.jpg
project: false
permalink: /blog/:title/
---

Plasma, a truly complete DE with everything you could ever dream of, you can
have it look like MacOS, windows 7, windows 10, and even Gnome(if you're into
that), the level of customizations KDE plasma provides is not matched by any OS
nor DE (quote me on that), even the things you think are not editable are in
fact editable (panel icons spacing).

The first time i installed plasma was back in 2017 when i got my student's
laptop and after waiting impatiently for budgie QT which never happened, used it
all the way until the beginning of 2019 at which i got bored and decided to tile,
enjoyed bspwm for the upcoming 7 months after that, even tho i wasn't using
plasma i was closely following up with their changelogs and updates especially
PointiestStick's [weekly posts](https://pointieststick.com/category/this-week-in-kde/),
the thing i noticed is a lot of the issues i was facing prior were fixed and
Wayland was coming up slowly but surely.

I decided to install plasma again just of boredom(exams are pretty boring) but i
had one issue, a significant one, since i have been tiling for the past few
months i simply can not go back to floating window managers and using one
desktops with windows on top of each other(like wut), so i had to find a way to
tile kwin, which i did before in the past and wasn't the best experience so i
thought why not just use bspwm, i then `yay -S plasma-meta` and didn't even
reboot a simple startplasma-x11 from inside my current terminal loaded plasma
with bspwm(i kid you not that actually worked!), with the installation out of
the way it was time to automate all this boring boot stuff.

### Setup
A good resource i found for using plasma with other window managers is their own
[guide](https://userbase.kde.org/Tutorials/Using_Other_Window_Managers_with_Plasma)
it helped me with settings for plasma floating windows, also gave me a basic
idea of how the boot process is going to work in this context.
The specific process was pretty easy using a script to set the `KDEWM` variable
which seemed redundant to me so i put that in my xprofile and it worked fine.
The next important thing was auto login and auto starting X after auto logging
it both of which were pretty easy (some changes to xprofile...etc.) since i
don't use a display manager

Here is my full xprofile
```bash
#!/bin/sh

# ~/.xprofile

# sourced at boot by ~/.xinitrc and display managers like lightdm

export XDG_CONFIG_HOME="$HOME/.config"
export KDEWM=/usr/bin/bspwm # this is the magic word

[ -d /sbin ] && PATH="$PATH:/sbin"
[ -d /usr/sbin ] && PATH="$PATH:/usr/sbin"
[ -d "$HOME/bin" ] && PATH="$HOME/bin:$PATH"

# super alone simulates Alt-F1
ksuperkey -e 'Super_L=Alt_L|F1' &
ksuperkey -e 'Super_R=Alt_L|F1' &

# dpms: timeout sleep off
xset dpms 600 900 1200

# keyboard repeat rate
xset r rate 350 60


 # autologin on tty1
if [ -z "$DISPLAY" ] && [ "$(fgconsole)" -eq 1 ]; then
exec startx
fi

```
### customizations

![plasma settings editor](/assets/images/plasma-settings.png)

With the boot process automated it was time to theme this thing (my favorite
part). Since i am using bspwm i can sxhkd with it so my keyboard shortcuts were
fine(just few adjustments to not conflict with plasma's own shortcuts) and
so were most of my bspwm settings. I cloned my good old
[plasma themes repo](https://github.com/Blacksuan19/Plasma-Themes) and starting
tinkering, to surprise there was now a built-in themes editor!! That made making
an oceanized plasma theme pretty easy, only had to change colors on Aex nomad
Dark. I edited a colorscheme as well and got everything to look like i am not
using plasma.

now for some tricks, the panel spacing always annoyed me since i was using it
and it seems that hasn't changed, not a big deal since changing that is easy,
just look for `spacing` in
```bash
/usr/share/plasma/plasmoids/org.kde.plasma.private.systemtray/contents/ui/main.qml
```
and you will find `spacing: 0` change that 0 to any number(i stick to 10 usually) and you're good.

### Benefits
there was no apparent reason why i tried this but there are many reasons why i
stayed. I made a list of things i have observed so far, there are probably more
stuff and some things that are not that important after all.

- Actually superior apps
> cant compare dolphin with any other file manager and cant compare plasma
settings with any other settings app, for me QT > GTK (quote me on that)
- Actually faster and cleaner
> maybe its just me but things seem faster especially app startups and
shutdown(no more annoying 1:30 min waiting for network manager since plasma
kills it automatically before shutdown)
- No CPU consuming polybar modules (thats the reason i tried this)
> if there was a reason for this switch it was probably this. plasmashell is
miles ahead.
- Many things are automatically configured
  - the widgets
  - sleep and hibernate
  - common shortcuts (audio, music controls)
  - no more fucked up time on wakeup (FUCK YEAH)
  - screen dimming
  - redshift night light
- Better notification system
  - interactive
  - download notifications
  - ability to mute notifications (auto when doing a presentation)
  - highly customizable
- Media widgets auto detects any player (even browsers!)
- Automatic dynamic bar icons
- I CAN HAVE THE AUDIO BE MORE THAN 100% WITHOUT FUCKING WITH PAMIXER(my speakers suck)
- Better theming and apps integration
  - gtk apps inside a QT environment are fine but the opposite is totally not true
  - both toolkits can be themed from one central (plasma settings)
  - arguably better looking fonts (noticed in chrome)


### how to integration
lastly, here are some good tips for anyone who wants to try this, just general
things that make sense.

#### bootup

- [auto login to tty1](https://wiki.archlinux.org/index.php/Getty#Automatic_login_to_virtual_console)
- export bspwm as the default WM from .xprofile(mentioned above)
- startx from xprofile(mentioned above)



#### using sxhkd with plasma shortcuts

- Benefits
  - much easier to configure than plasma settings
  - a single file that can be tracked
  - no GUI needed (cuz meh that)
- Keep the general bspwm shortcuts and the custom apps
- Disabling the conflicting stuff
  - audio
  - media switches
  - brightness
  - screen shot(if you want spectacle instead of flameshot, why tho?)

#### theming

my suggestion is to eliminate GTK apps as much as possible, that will make this
kinda easier.

- `kde-gtk-config` package for gtk apps settings
- Making shit even cleaner
  - [better icons](https://github.com/keeferrourke/la-capitaine-icon-theme)
  - [unified theming](https://github.com/material-ocean/Material-Ocean)

#### current issues

- Cannot right click on desktop (who needs that anyway)
- Plasma floating windows can steal focus(kinda annoying sometimes)

#### misc stuff

- Use flags for language representation
- [better virtual desktops widget](https://github.com/wsdfhjxc/virtual-desktop-bar)
- [dotfiles](https://github.com/Blacksuan19/Dotfiles) for more stuff

### Conclusion

This was a pretty good experience and it works much better than i expected it,
the amount of progress plasma made in those 7-8 months i wasn't using it is
astonishing and makes me look foreword to plasma in 2020, this is gonna be my
setup for this year and maybe after, only major change i might consider is going
the Wayland way that is only if there is a good bspwm alternative over there.


Now here is a workflow video, enjoy!
{% include YouTube.html id="lZbk-QEqSTM" %}

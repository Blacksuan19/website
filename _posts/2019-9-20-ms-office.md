---
layout: post
title: Microsoft Office On Linux
description: How to get full office formats support on linux
image: /assets/images/office.jpg
project: false
permalink: /blog/:title/
---
The semester just started and like other semesters and other universities in this whole planet they use proprietary microsoft formats to distribute study material like slides and documents, why cant you just use regular formats that work everywhere (like pdf)!! <br><br>
Last sem most of the materials didn't have compatibility issues with the office apps i use ([WPS office](https://www.wps.com/index.html){:target="_blank"} and recently [Only Office](https://www.onlyoffice.com/){:target="_blank"}), now people are gonna jump with the stupid ```wHy cAnT yOu uSe LiBrEOfFiCe```, tbh screw libreoffice, it looks like trash, its heavy and its slow af to open not to mention still not fully compatible with with proprietary office files. <br><br> So i use the other 2 and each has their own good features, only office is just one window with docs, sheets and slides which i really love, i can switch back and foreword without having to switch windows (which is arguably not that intuitive on a tiling wm), wps office on the other hand has better compatibility and renders slides that look exactly like on office (only office has issues with black on black text).<br><br>
Anyway, lets not complain and just fix the issue and get to the point, as a linux user and someone who despises microsoft and their shit of an OS which i really don't want to get into its issues and how they basically monopolized the OS space and took control of everything and made their shitty software the go to standard. <br><br>
The new Wps office has a neat feature which checks if a document has missing fonts and shows you those fonts and what its going to replace them with, those replacements are usually not that good (this thing replaced MS Gothic with weather icons like bruh) so you have to get that font and install it.<br> <br>
#### The main event:
- Download and install [WPS office](https://www.wps.com/index.html){:target="_blank"} or install [from aur](https://aur.archlinux.org/packages/wps-office/){:target="_blank"} if you're on arch
- Download the full microsoft fonts folder [from here](https://drive.google.com/open?id=1UlIQRj837nHI7FgqEjzFy7wctaiR9Z3F){:target="_blank"} (yeah i pirated that basically, pls MS don't take this site down)
- you might need to update your font cache ```fc-cache -f -v```
- copy all the fonts to ```~/.fonts```
- profit!

<img src="/assets/images/wps.png">
<br><br><br>
and with this all the documents i tried had full fonts and worked perfectly with no issues, if you face issues just check the missing fonts and you will probably find them online somewhere. lastly, i chose wps office (other than the font checker) because it also does load your fonts, onlyoffice has a font folder that it loads from and it doesn't have not even half of the fonts included here.<br><br>
With what microsoft is doing right now for linux i hope they'll bridge this gap one day and get us a full office suit that doesn't need any hacks  or emulation to function properly.
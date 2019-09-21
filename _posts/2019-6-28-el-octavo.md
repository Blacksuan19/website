---
layout: post
title: Dark Ages El Octavo
description: the 8th release of dark ages
image: /assets/images/da.jpg
project: false
---

today marks a whole year since the first DA release, and issa a special day, it has come a long way, from the good stuff to the great stuff to the weird bugs caused by firmware's and even more weird bugs caused by what we still deem as unknown.<br>

this release also marks the first release based on the 4.9 kernel source, huge thanks to the people who made that possible (rama, luan, flex, mdeejay and others), without those nibbas el octavo would've still been 3.18 which sucks at this time where everything is moving towered the 4.9 route, also not to forget ma nibba amulya for making good trees (even tho he derped some stuff) cuz without device trees you don't have roms which means you cant flash custom kernels (don't even think about mIUi CuStOm KeRnUl).<br>

finally, with this release the goal is mainly to deliver a stable UX, and try to avoid adding unnecessery features as much as possible and also to intreduce the new ramdisk embedded profiles which makes it very easy to apply custom settings on every boot, i managed to get them to work even tho i broke init multiple times, the new kanged smurf governor gives a very good experience and its as scalable as governor could ever be i would even say its better than electron cuz its based on it and others (darkutil), oh yeah and since this is 4.9 now we have EAS (energy aware scheduling) and we have a working energy model which is noice.

###### Default profile:

the default profile as mentioned before leans towered the balanced side, i personally don't play games and find them a waste of time so don't expect a gaming profile from me.

> governor: pixel_smurfutils (tweaked)<br>
> Min CPU frequency: 652<br>
> cpu stune boost: on (tweaked)<br>
> read_ahed: 512kb<br>
> dynamic fsync: on<br>
> usb fast charge: on<br>
> tcp: westwood<br>

###### Battery life:

some might say the battery took a hit with 4.9, but i partially agree, i see some idle drain which was a thing since nougat, but its not much of a big deal, with EAS the kernel should be able to better manage battery consumptions (i hope so).

<div class="row 200%">
    <div class="6u 12u$(medium)">
    <img src="/assets/images/bat1.jpg">
    </div>
    <div class="6u 12u$(medium)">
    <img src="/assets/images/bat2.jpg">
    </div>
</div>
as you can see those are solid af stats (which also uses material ocean theme btw), even tho the drain doesn't corelate with SOT which is what we call idle drain.<br>

for full changelog and Downloads check the downlaods Page

<ul class="actions">
    <li>
        <a href="{{ site.url }}/da" class="button special fit">Right here</a>
    </li>
</ul>

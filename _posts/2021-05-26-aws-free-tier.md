---
layout: post
title: Don't Be Fooled By AWS free tier
description: it's a trap
image: /assets/images/aws.jpg
project: false
permalink: /blog/:title/
tags:
  - cloud
  - machine-learning
  - data-science
---

Amazon web services, one of the big 3 cloud services provider alongside GCP and
Azure, AWS provides a diverse range of cloud services from basic servers and
databases to deep learning specialized hardware and cloud notebooks, I have
personally used all 3 providers and found GCP to be the easiest, azure to be the
most boring and AWS to be the most complex, any task requires a good amount of
time to prepare resources such as IAM roles (which are even more complex
compared to GCP) and start, it also has a free tier that offers basic servers
and notebook instances for a starter.

## The problem

when there is a system this complex there will be problems eventually, there
will be services not used because people were not able to figure it out, the
documentations do an adequate job of explaining some processes but some
unconventional workflows have no documentation at all whatsoever (deploying a
model with multiple pre-trained networks).

when the free tier is over you would expect AWS to email you or something to
notify you about it but no that's not the case, not only don't they not message
you they also start to rack up fees that could get to 100s of dollars.

- no notification when free tier is finished.
- no notification when a service that is part of the free tier starts to
  accumulate charges.
- unconventional way of handling notebooks, on other cloud services when a
  notebook page is closed the notebook is automatically shut down, that's not
  the case in AWS, it could stay running for months without any warning. their
  reasoning behind this is users have the choice to shut it down manually
  combined with the other 2 above points this is a financial disaster waiting to
  happen.
- there is no easy way to delete all resources related to a task, for instance I
  created a sagemaker studio instance for experimenting, when I was done with it
  and deleted it I thought that's it but apparently that sagemaker instance
  created a separate disk for its data that is not deleted when the instance
  itself is deleted and now it has accumulated extra unnecessary charges for
  staying on even when its not connected to anything or being used anywhere.
  only indicator to the disk's existence is the bill.

### cases of "free tier" being billed

there are many cases in many different websites, from reddit, twitter and
personal blogs to even AWS forums, I found a thread from 2011 talking about this
same issue and mentioning similar solutions that had some discussion and went
unnoticed.

- [NiekHoupelyne twitter](https://twitter.com/NiekHoupelyne/status/1306207091883995136)
- [Andrew Ray's blog](https://kutt.it/sEyfIi)
- [Reddit](https://www.reddit.com/r/aws/comments/btvlx2/aws_sagemaker_charged_5000usd_for_notebook/)
- [Reddit 2](https://www.reddit.com/r/aws/comments/3zoia7/help_aws_charged_me_165_usd_for_nothing_and/)

### "why not set a budget?"

most of the cases where an unexpected bill is delivered is from free tier users,
by the definition of it being "free" no one will check the billing section and
there is no annoying (but useful) banner that shows your free remaining credit
like in GCP.

## Solutions

it would be wrong to mention issues without giving solutions so here are some

- notify users when the free tier is over.
- notify users when a service starts to accumulate charges.
- auto delete automatically created resources when a service is terminated,
  expecting users to be not just aware of a service they did not create but to
  also delete it/shut it down is plain wrong and looks predatory.

## My experience

i started using AWS in the beginning of 2021 to work on my final year project, i
needed a powerful notebook with graphical resources, in the beginning it was
very easy to create the sagemaker studio instance and start coding, I started
with the `ml.t2.medium` instance which is free, did some testing without running
any training or inference.

after that I left AWS for around 2 months because I was busy with other things
and found training on kaggle to be better and is free!, in march I tried to
deploy my first endpoint which did not go well, my model was a combination of 3
pre-trained networks that run one after one, I googled around, read the docs and
never found anything about deploying this type of model being possible, I gave
up on AWS and decided to deploy somewhere else.

April is when things start to go down, my bill was suddenly $84 so I checked my
billing and found sagemaker to be the culprit, apparently my "free tier" ended
the month prior and from April AWS started charging me for my `ml.t2.medium` and
for storage, all this happened without any notification, the only email i
received was about S3 doing too many requests which I was aware of, not to
mention it only accounted for $0.73 of my bill. I only noticed this charge at
the beginning of may and raised a ticket which did not get me anywhere and they
of course pointed all fingers at me.

![april bill](/assets/images/bill1.png)

in the month of May I was done with my project and wanted to test the training
scripts on an environment other than a notebook, so I fired up a `ml.t2.medium`
instance (i have deleted all prior instances after the unexpected bill), my
dataset was around 60GB which I thought would be covered without any charges
(yeah stupid I know) so I started downloading it, after testing everything i
made sure to delete the whole instance, fast forward to around mid-May I checked
my bill and found that I have to pay $17 for something called EFS, my first
thought was what in the hell is EFS? and when did I create it? I looked around
as usual and found an EFS instance running even though I do not recall creating
it, after deleting it I raised that issue in a ticket and as usual they pointed
the fingers at me, even when I mentioned I did not even know what EFS is let
alone create an instance there, all the replies just included their "agreement"
quoting the part about the user being responsible of everything done with their
account even if it wasn't them who did it.

![may bill](/assets/images/bill2.png)

they kept sending me the agreement as a response for all my questions and
explanations, I figured this was going no where and I was not getting my money
back or the EFS charges dropped but I still did not close the ticket, after a
while they closed the ticket with a "thank you".

billing issues aside, setting up an S3 bucket where everything is publicly
accessible took longer than I thought it should, I have around 200,000 images
that I want to be publicly accessible for an API, I had to make the folders
public which does not make their content public (???) so I had to write a script
that goes through all files in a folder and makes them public, that took a while
to finish but at least it worked, I thought my issues were over but again sike,
CORS was bugging as CORS be bugging all the time, found a stackoverflow question
where they just enabled access to all origins so I did that, another crisis
averted.

## Conclusion

i wrote this article hoping to enlighten new free tier users about the scam
called AWS free tier, because I was that stupid and went through it already, i
have learned my lesson and set up budget alerts while also making sure to check
if there are any extra services being automatically created when I create an
instance of something. in conclusion the free tier is not free and is a way to
get unknowing users to make Jeff even richer. I will close with this quote from
the editor of last week in AWS

> The AWS Free Tier is free in the same way that a table saw is childproof. If
> you blindly rush in to use an AWS service with the expectation that you won’t
> be charged, you’re likely to lose a hand in the process.

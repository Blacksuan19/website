---
title: VirusTotal Bulk Checker
layout: post
description: Automating the Boring stuff with python
image: "/assets/images/virustotal.png"
project: false
permalink: "/blog/:title/"
tags:
- python
- auto
- virustotal
- API
---

VirusTotal, a truly great service, making it easier for anyone to check any
file, hash or URL against multiple antivirus databases in a little to no time,
their API is also an absolute gold, with it you don't need to deal with  browser
GUI, you can make simple requests and get results in most programming languages,
hell you could even check a file using just curl or even wget, the only drawback
is that the [public API](https://developers.virustotal.com/reference) has a limit
of 5 checks per minute (maybe its less?), after which you
need to wait another minute to check anything again.

### Scenario

I have a file full of hashes (~2000) and i need to check all of the hashes with
virus total, doing this on the browser is very tedious and time consuming, time
in which you could be doing something else more productive or just chilling.

### Available solutions

The [API scripts page](https://support.virustotal.com/hc/en-us/articles/115002146469-API-Scripts)
has examples of implementations in different languages and non of
which supports bulk checking and or even reads hashes from a list. That leaves
me hanging and in search of a solution.

### Building the script

It is very simple to deal with the API, you send it a file, hash or a URL and it returns
a json file with the results of the scan, the json file contains the number of
engines detecting the file as malicious and list of all their 56 antiviruses,
the json file also contains link to the report which we don't really need but
worth mentioning.

#### Preparing the file
before we can scan the files we need to put them in a way that we can traverse
them and that sounds like a simple list. We also need another file to which we
can send the results or at least a simple summery if the results.

```python

hashes = open("hash value.txt") # the hashes to check
analysis = open("analysis.txt", "w") # the file to save to the result
analysis.write("\t\t\tHash\t\t\t\t\t# of engines"
               "detected\n========================================================\n\n")
```
in this case we are going to print the hash and the number of engines detected
it as malicious, just to keep it simple.

For dealing with the API, we need an API key which you can easily obtain by signing up
we send a request with the hash to `https://www.virustotal.com/vtapi/v2/file/report`
which replays with status 200 and sends back the json file if all is fine.

Now we need to traverse all hashes and appropriately pass them one by one to the
api to get some results!

```python
for hashn in hashes:
      print('Checking hash ' + hashn)
      params = {'apikey': apikey, 'resource': hashn}
      response = requests.post('https://www.virustotal.com/vtapi/v2/file/report',
                                params=params)
      result = response.json()

```

#### Checking the hashes
Now we have the response as a json file, one of the reasons i chose python for
this (beside it being easy and currently on my learning spectrum)
is because you can easily manipulate json files without needing any hacks
or extra libraries

The `positives` filed in the json indicates the number of positives, that makes
it super easy to just grab that, but grabbing all files would just bloat the
output file and add unnecessary data, so lets first check if it is not 0 (no
matches)

```python

# write only the files recognized as malicious
if result['positives'] != 0:
      analysis.write(hashn.strip() + "\t\t\t" +
                    str(result['positives']) + "\n")

```

#### Going around the limit
And that is basically all, we have a functional script that does what its
supposed to and saves us time, except not really since it still just crashes on
the 5th check (the public API limit), to counter this we can make the script
wait 1 minute if it encounters an error while scanning, the best way to do that
is not by using a limit variable (pretty stupid), its rather just throw the
whole thing in a try-except block, if it errors thats an exception so we wait

```python

except:
    print("API limit reached!, lets stall...")
    time.sleep(1 * 60)

```


### Final results

And now we are done, although this checks 5 hashes every minute which means it is
going to do 1000 hashes in 200 minutes (3.3 hours) adding the sleep time means
it will take 6.6 hours in total, not too bad if you don't really need the data
right now, a better option would be just to get the private API and break the
limit.

The final script

```python

import requests
import json
import time

# check a file of hashes with virustotal and save results

# get you api key
apikey = 'YOUR_API_KEY'
hashes = open("hash value.txt") # the hashes to check
analysis = open("analysis.txt", "w") # the file to save to the result
analysis.write("\t\t\tHash\t\t\t\t\t# of engines"
               "detected\n========================================================\n\n")

for hashn in hashes:
    try:
        print('Checking hash ' + hashn)
        params = {'apikey': apikey, 'resource': hashn}
        response = requests.post('https://www.virustotal.com/vtapi/v2/file/report',
                                params=params)
        result = response.json()

        # write only the files recognized as malicious
        if result['positives'] != 0:
            analysis.write(hashn.strip() + "\t\t\t" +
                               str(result['positives']) + "\n")
    except:
        print("API limit reached!, lets stall...")
        time.sleep(1 * 60)

```

The outputted file is just as it should be, simple and informative, it only
shows the hash and number of engines detected. That is of course filtering to
only the files which are malicious.

```bash
			Hash					# of engines detected
========================================================

004d3a700ba37a463e4a4b1ab452651a			45
0136b7453cedf600d6f6aab7900901d3			36
027d2b35e10c8e9a20cd3ca7112d12fb			42
03375389a0bba9e96e58f6c68880f15c			41
0418b65d9ab22e75740bb30c42221abb			42
04586aa35ab1eae9720cb36a216f0ef4			42
09543abb894b7009ee509d978cb35b85			43
0ac73aa8c95fe19eae64e0e130ca1d9a			40
0d4781b799433fa23547c87d67c8e36b			43
0e2cefa612a12dfe30b3782d9bc53089			44
0fedb99715db80729d2887d090bed329			38
11ac382313515f7b64b27191b4f90dbd			45
14a6bf827b5eb322c068794047a2b56f			44
14f026826540eada355aa39a97ccc9d8			44
15554b42b7405bf26514add6fe735025			45
158300595c7aa5e711cea5039b48ced9			45

```

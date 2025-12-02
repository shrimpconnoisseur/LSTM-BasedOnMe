# LSTM-BasedOnMe
I created and trained a tiny language model on all my Discord DMs to friends.
As expected, it's dumb and only outputs gibberish.
However, I guess you can kind of tell what kind of person I am and what I talk about through this thing. 
It may or may not output slurs.

If you want to run it, open up the terminal:
`python lstm.py --test`

If you want to train it for some reason, it takes in a JSON file in the format:
`{
    "ID": 0101010,
    "Timestamp": "2022-02-25 23:34:21",
    "Contents": "something is said here",
    "Attachments": ""
}`
Honestly, you really only need the "Contents" line.

Then execute in a terminal:
`python lstm.py --train --json messages.json`

The training phase will know to only read from the "Contents" line.
It took me two hours to train this thing for such a garbage result.
Probably because I didn't know what a good learning rate was, because the training dataset was way too small, or because I really do just speak like a schizophrenic.

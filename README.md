# BangleBeat

A machine learning approach to improving Bangle.JS 2 heart rate estimations for an individual.

## Overview

BangleBeat is an attempt to improve heart rate estimations from an inexpensive smart watch, by learning patterns from a more accurate device. The process involves building a correction model based on tens of thousands of heart rate measurements from each device. The notobook then distills the model down into a ready-to-use Javascript application that is ready to deploy on the Bangle JS 2, and provide on-device ML heart rate corrections.

This approach hinges on historic heart patterns accurately predicting future heart patterns. While this is generally accepted, the accuracy will drift with a person's age, fitness level, and other factors that I will call a "health profile" for the sake of brevity. The assumption, then, is that the subject has a reasonably identical health profile to the __entire dataset__. Long term use of this kind of approach should realistically remove old or irrelevant data as a person's health profile changes.

**Disclaimer:** This project is a learning exercise in embedded AI/ML and does not claim to be cutting edge heart rate measurement research. This work is in no way intented or implied to be appropriate for medical use.

!["Component Diagram"](assets/comoponent-diagram.svg)


## Prerequisites

- [Bangle JS 2](https://shop.espruino.com/banglejs2)
 
- At least one reference heart rate measuring device (I chose a Polar H10 chest strap and a Garmin Instinct 2X watch). Just make sure it is [supported by Gadgetbridge](https://gadgetbridge.org/gadgets/)! 

- An android phone (to run gadgetbridge)


## Getting Started

1. Install [Gadgetbridge](https://gadgetbridge.org/) on your Android phone.

2. Connect your reference device(s) to Gadgetbridge.

3. Install [heart](https://banglejs.com/apps/?id=heart) on your Bangle JS 2.

4. Log some data! See `Data Collection Strategy` below for some tips.

5. Set your environment up! This repository uses [nix](https://nixos.org) to make this work reproducible. If you're unfamiliar with what that means don't worry at all -- just use the OCI approach.

**Heads up: nix isn't light on storage. This will take up ~6GB on your system. Plan accordingly!**

<details><summary> nix (unix-like only)</summary>

```bash
# (In this cloned repo, of course)
# builds the development environment, enters a shell with all required libraries and tools
nix develop 
```
</details>

<details><summary> OCI (Podman/Docker, Windows users go with this one)</summary>

Podman and Docker are more or less interchangeable. Run with whichever you have installed.

Podman: 

```bash
# initializes a container nix environment
podman build -t banglebeat .                       
# mounts the directory in the container, then runs `nix develop`
podman run --rm -ti -v .:/src localhost/banglebeat 
```

**or** Docker: 

```bash
# initializes a container nix environment
docker build -t banglebeat .                       
# mounts the directory in the container, then runs `nix develop`
docker run --rm -ti -v .:/src localhost/banglebeat 
```

</details>

6. Copy your Gadgetbridge DB and into the `./data` directory and run the notebook.


## Data Collection Strategy

Things to know:

1. Heart rate sensing is an *expensive* process for a little embedded computer. I was anecdotally able to log data with my BangleJS for ~24h but your mileage may vary. More samples means more power consumption.

2. Choose a great reference device for you. ECG chest straps will give you very accurate data, but might not be comfortable to wear for long periods of time. Wrist measurement devices can be very good, but it can be a biomechanically challenging place to measure heart rate data from. Rings, in-ear headphones, and other devices may do the job too. Garbage in equals Garbage out, so be careful about the data you choose.

3. Gadgetbridge, while an excellent project, might not give you immaculate data for a purpose it wasn't strictly built for. You might need to tinker with some settings to make sure you're getting _as much data as possible_. 

4. The biggest hurdle will likely not be if you have data, but if you have _intersecting_ data. It might feel a little goofy wearing 2-3 heart rate sensors at once, but it's vital if you want good results with this approach.

5. I had a lot of luck with [Gadgetbridge automated exports](https://gadgetbridge.org/internals/development/data-management/) and [Syncthing](https://syncthing.net/) to automatically copy the data back to my computer. Once per day seemed fine.

6. Accumulating data takes time! A week of collection should be enough to get a usable model. However, you can always keep piling up data and making incremental improvements! The more intersecting data you have the better your corrections (guesses) will be.

## Future work

I plan to take continue this work in a few ways. 

Short term: port data loading and ML training from `main.ipynb` into [the web interface for my experimental deployment app](https://github.com/lucspec/BangleApps/tree/master/apps/hrdynamics/interface.html)

Medium term: merge my model deployment app ([hrdynamics](https://github.com/lucspec/BangleApps/tree/master/apps/hrdynamics)) into upstream so others can use this approach to train + deploy their own models.

Long term: merge this functionality with [Gadgetbridge](https://codeberg.org/Freeyourgadget/Gadgetbridge) for continuous on-device, always private, personal model training. Time will tell if this aligns well enough with Gadgetbridge's goals to merge into upstream or if I will need to maintain it as a fork.

## Recognition

- [all of the contributors to bangle apps `heart`](https://github.com/espruino/BangleApps/commits/master/apps/heart/app.js) for making the base data collection possible

- Gadgetbridge maintainers; especially who implemented device support for the devices used in this work!


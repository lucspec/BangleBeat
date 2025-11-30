# BangleBeat

A machine learning approach to improving Bangle.JS 2 heart rate data


## Prerequisites

- [Bangle JS 2](https://shop.espruino.com/banglejs2)
 
- A reference device [supported by Gadgetbridge](https://gadgetbridge.org/gadgets/); 

- An android phone


## Getting Started

1. Install [Gadgetbridge](https://gadgetbridge.org/) on your Android phone.

2. Connect your reference device(s) to Gadgetbridge.

3. Install [heart](https://banglejs.com/apps/?id=heart) on your Bangle JS 2.

4. Log some data! See `Data Collection Strategy` below for some tips.

5. Set your environment up! This repository uses `Nix` to make this work reproducible. If you're unfamiliar with [nix](https://nixos.org/) I recommend using the OCI approach instead of trying to install nix just for this project.

**Warning: Nix isn't light on storage. This will take up ~6GB on your system. Plan accordingly!**

<details><summary> Nix</summary>

```
# builds the development environment, enters a shell with all required libraries and tools
nix develop 
```
</details>

<details><summary> OCI (Podman / Docker)</summary>

```
# initializes a container nix environment
podman build -t banglebeat .                       
# mounts the directory in the container, then runs `nix develop`
podman run --rm -ti -v .:/src localhost/banglebeat 
```
</details>

6. Copy your Gadgetbridge DB and into the `./data` directory and run the notebook.


## Data Collection Strategy

Things to know:

1. Heart rate sensing is an *expensive* process for a little embedded computer. I was anecdotally able to run my BangleJS with [heart](https://banglejs.com/apps/?id=heart) logging on for ~24h but your mileage may vary.

2. Gadgetbridge, while an excellent project, might not give you immaculate data for a purpose it wasn't strictly built for. You might need to tinker with some settings to make sure you're getting _as much data as possible_. 

3. The biggest hurdle will likely not be if you have data, but if you have _intersecting_ data. It might feel a little goofy wearing 2-3 heart rate sensors at once, but it's vital if you want good results with this approach.

4. I had a lot of luck with [Gadgetbridge automated exports](https://gadgetbridge.org/internals/development/data-management/) and [Syncthing](https://syncthing.net/) to automatically copy the data back to my computer. Once per day seemed fine.

5. Accumulating data takes time! A week of collection should be enough to get a usable model. However, you can always keep piling up data and making incremental improvements! The more intersecting data you have the better your corrections (guesses) will be.

## Future work

I plan to take continue this work in a few ways. 

Short term: port data loading and ML training from `main.ipynb` into [the web interface for my experimental deployment app](https://github.com/lucspec/BangleApps/tree/master/apps/hrdynamics/interface.html)

Medium term: merge my model deployment app [hrdynamics](https://github.com/lucspec/BangleApps/tree/master/apps/hrdynamics) into upstream so others can use this approach to train + deploy their own models.

Long term: merge this functionality with [Gadgetbridge](https://codeberg.org/Freeyourgadget/Gadgetbridge) for continuous on-device, always private, personal model training. Time will tell if this aligns well enough with Gadgetbridge's goals to merge into upstream or if I will need to maintain it as a fork.

## Recognition

- [all of the contributors to bangle apps `heart`](https://github.com/espruino/BangleApps/commits/master/apps/heart/app.js) for making the base data collection possible

- Gadgetbridge maintainers; especially who implemented device support for the devices used in this work!


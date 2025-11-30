# BangleBeat

A machine learning approach to improving Bangle.JS 2 heart rate data

## Setup

This repository uses `Nix` to make this work reproducible. If you don't use `Nix` -- simply use the `Containerfile` with an OCI tool of your choice (likely podman or docker). 

**Warning: Nix isn't light on storage. This will take up ~6GB on your system regardless of the method you choose. Plan accordingly :)**

### Nix

```
# builds the development environment, enters a shell with all libs and tools
nix develop 
```

### OCI (Podman / Docker)

```
# initializes a container nix environment
podman build -t banglebeat .                       
# mounts the directory in the container, then runs `nix develop`
podman run --rm -ti -v .:/src localhost/banglebeat 
```


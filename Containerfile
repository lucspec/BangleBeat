# Use the official Nix image based on Alpine
FROM nixos/nix:latest

# Configure Nix to enable flakes and nix-command
# We append to the existing configuration file
RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf

COPY . /src
WORKDIR /src

# By default, drop into the nix development shell
# IMPORTANT: You must run docker with '-it' for this to work
CMD ["nix", "develop"]

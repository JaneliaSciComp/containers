# Containers

Docker (OCI) containers for reusable workflow plugins. These are used in [Nextflow Modules](https://github.com/JaneliaSciComp/nextflow-modules). 

# Principles

This repository borrows heavily from [BioContainers](https://github.com/BioContainers/containers) effort. 

We want containers that:

1) Are reproducible - if you rebuild the same Dockerfile a year later, you should get basically the same container. 
2) Are optimized - We use standard techniques to speed up builds and we use multistage builds to reduce the size of the final image. 
3) Include metadata -  We currently use [BioContainer metadata](https://github.com/BioContainers/specs/blob/master/container-specs.md)


# Publishing images

Containers in this repository are published as packages in the  GitHub Container Registry using these steps: 

1. Obtain a GitHub [personal authentication token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) and save it into your terminal environment as `$GITHUB_PACKAGE_TOKEN`. 
   
2. Log into the registry

```bash
echo $GITHUB_PACKAGE_TOKEN | docker login ghcr.io --username rokickik --password-stdin
```

3. Build your image

```bash
docker build stitching-spark/1.10.0 -t ghcr.io/janeliascicomp/stitching-spark:1.10.0
```

4. Push your image

```bash
docker push ghcr.io/janeliascicomp/stitching-spark:1.10.0
```



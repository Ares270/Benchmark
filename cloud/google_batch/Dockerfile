FROM --platform=linux/amd64 mambaorg/micromamba:2.8.1@sha256:79284aa2949ac9555eca7e975ceb4ecefefc5964a2d4fdd4e181ca8eaccf347e

COPY --chown=$MAMBA_USER:$MAMBA_USER cloud/google_batch/environment.yml /tmp/environment.yml
RUN micromamba install --yes --name base --file /tmp/environment.yml \
    && micromamba clean --all --yes

WORKDIR /opt/benchmark
COPY --chown=$MAMBA_USER:$MAMBA_USER src /opt/benchmark/src
COPY --chown=$MAMBA_USER:$MAMBA_USER cloud/google_batch/run_chunk.sh /opt/benchmark/cloud/google_batch/run_chunk.sh
COPY --chown=$MAMBA_USER:$MAMBA_USER data/target/7O7K_protein.pdbqt /opt/benchmark/data/target/7O7K_protein.pdbqt

ENV PYTHONPATH=/opt/benchmark
ENV PYTHONDONTWRITEBYTECODE=1

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python -c "from rdkit import Chem; from meeko import MoleculePreparation; print(Chem.__version__ if hasattr(Chem, '__version__') else 'RDKit import OK'); print(MoleculePreparation)" \
    && smina --version

USER root
CMD ["/bin/bash", "/opt/benchmark/cloud/google_batch/run_chunk.sh"]

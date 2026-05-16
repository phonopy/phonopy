(workflow)=

# Work flow

## Phonon calculations at constant volume

Work flow of phonopy is shown schematically. There are two ways to calculate,
(1) atomic forces from finite displacements and (2) given force constants. You
can choose one of them. Forces on atoms or force constants are calculated by
your favorite calculator (shown by the octagons in the work flow). The boxes are
jobs being done by phonopy, and the circles are input and intermediate output
data structures.

```{mermaid}
flowchart TD
    UC(["Unit cell"])
    SS(["Supercell size"])
    PCS(["Primitive cell size<br/>(auto)"])
    NAC(["Non-analytical term<br/>correction parameters<br/>(recommended)"])
    NACOPT(["Non-analytical term<br/>correction parameters<br/>(optional)"])
    DISP(["Displacements"])
    SC(["Supercell"])
    FC(["Force constants"])

    PRE["phonopy-init"]
    POST["phonopy"]

    FCALC{{"Force calc."}}
    FCCALC{{"Force-constant calc."}}

    UC --> PRE
    SS --> PRE
    PCS --> PRE
    NAC --> PRE
    PRE --> DISP
    PRE --> SC
    DISP -->|"(1)"| FCALC
    SC -->|"(1)"| FCALC
    SC -->|"(2)"| FCCALC
    FCALC -->|"(1)"| FC
    FCCALC -->|"(2)"| FC

    UC --> POST
    SS --> POST
    FC --> POST
    NACOPT --> POST

    POST --> BS["Band structure"]
    POST --> MS["Mesh sampling"]
    POST --> QP["Specific q-point"]

    MS --> DOS["DOS"]
    MS --> PDOS["PDOS"]
    MS --> TP["Thermal properties"]
    MS --> MSD["Mean square displacement"]

    MSD --> DSF["Dynamic structure factor"]
    QP --> DSF
    QP --> AM["Atomic modulations"]
    QP --> IR["Irreducible reps."]

    classDef init fill:#dae8fc,stroke:#6c8ebf,color:#000
    classDef run fill:#d5e8d4,stroke:#82b366,color:#000
    classDef ext fill:#ffe6cc,stroke:#d79b00,color:#000
    class PRE init
    class POST run
    class FCALC,FCCALC ext
```

The blue box is the `phonopy-init` setup step, the green box is the
`phonopy` post-process (phonon calculation), and the orange hexagons
are external force calculators. Path (1) goes through a force
calculator that returns atomic forces; path (2) goes through a code
that returns force constants directly (e.g. VASP-DFPT).

## Combinations of phonon calculations at different volumes

Mode Grüneisen parameters can be calculated from two or three phonon calculation
results obtained at slightly different volume points. See the details at
{ref}`phonopy_gruneisen`.

With more volume points and fitting the thermal properties, thermal properties
at constant pressure are obtained under the (so-called) quasi-harmonic
approximation. See more details at {ref}`phonopy_qha`.

# The Visual Construction of the Enemy

**Cultural Analytics Project – Winter Term 2025/26**
**Leipzig University**

A quantitative analysis of "Us" vs. "Them" visual representation in Nazi-era films.

---

## Research Question

Which visual features did Nazi film production systematically employ to distinguish the ingroup ("Us") from the outgroup ("Them")? Can these differences be learned automatically, and which visual features are most relevant for this distinction?

## Focus

The analysis focuses on the ideological and "racial" primary enemies of Nazi ideology, particularly the concept of "Jewish Bolshevism" (Jews, Communists, Soviets), rather than the depiction of Western Allies. Hitler maintained an ambivalent relationship with the British in particular, viewing them as a "brother nation."

## Methodology

- Extraction of visual features (lighting, camera angle, shot scale, composition, pose) from annotated frames
- Training of a classical ML classifier (e.g., Random Forest) to distinguish "Us" from "Them"
- Feature importance analysis to identify which visual characteristics propaganda most strongly employed for enemy image construction

---

## Dataset

**Source:** [Internet Archive – German Films 1933-1945](https://archive.org/details/movies?and%5B%5D=mediatype%3A%22movies%22&and%5B%5D=language%3A%22German%22&and%5B%5D=year%3A%5B1933+TO+1945%5D)

### Enemy Image Films ("Them" – Outgroup)

**Antisemitic Films:**
- *Der Ewige Jude / The Eternal Jew* (1940) – "Documentary," explicit antisemitic propaganda
- *Jud Süß / Jew Süss* (1940) – Feature film, antisemitic stereotype of the "power-hungry Jew"
- *Die Rothschilds / The Rothschilds* (1940) – Feature film, enemy image of "Jewish finance capital"

**Anti-Bolshevik/Anti-Communist Films:**
- *Hitlerjunge Quex / Hitler Youth Quex* (1933) – Communists as enemies of German youth
- *Hans Westmar* (1933) – SA martyr vs. Communists
- *GPU* (1942) – Soviet secret service as enemy image
- *Flüchtlinge / Refugees* (1933) – Volga Germans vs. Bolsheviks

**Anti-Polish/Anti-Slavic Films:**
- *Heimkehr / Homecoming* (1941) – "Ethnic Germans" as victims of Polish violence
- *Feinde / Enemies* (1940) – German minority in Poland

### Ingroup Films ("Us" – Volksgemeinschaft)

**Explicit Propaganda:**
- *Triumph des Willens / Triumph of the Will* (1935) – Nazi Party Rally, idealized "national community"
- *Olympia* (1938) – "Aryan" body ideals
- *Kolberg* (1945) – Perseverance propaganda, German heroes

**Entertainment Films with Positive Ingroup Representation:**
- *Die große Liebe / The Great Love* (1942) – German soldiers and home front
- *Wunschkonzert / Request Concert* (1940) – National community on the home front
- *Stukas* (1941) – Heroic Luftwaffe pilots

---

## Project Structure

```
├── data/
│   ├── raw/                # Extracted frames
│   ├── annotated/          # Annotated frames with labels
│   └── features/           # Extracted features
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_annotation.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_analysis.ipynb
├── src/
│   ├── feature_extraction.py
│   ├── model.py
│   └── utils.py
├── results/
│   ├── figures/
│   └── tables/
├── README.md
└── requirements.txt
```

---

## Authors

| | Kevin Kunkel | Felix Filius |
|---|---|---|
| **Program** | M.Sc. Computer Science | M.Sc. Data Science |
| **E-Mail** | [email] | [email] |
| **Student ID** | [student ID] | [student ID] |

---

## License

This project was created as part of the "Cultural Analytics" module at Leipzig University.

## Disclaimer

The films analyzed in this project contain National Socialist propaganda and antisemitic content. This analysis serves exclusively academic purposes in the context of historical research and education.
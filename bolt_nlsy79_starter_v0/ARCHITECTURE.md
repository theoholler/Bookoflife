# Architecture

Two notebooks share the reshape + prediction halves; they differ only
in **how the Book of Life is rendered**.

- `bol_pipeline_simplified.ipynb` — pure pandas; reads the processed
  CSVs directly and writes `books.json`. No DuckDB, no recipes, no
  `Paragraph` classes.
- `bol_pipeline.ipynb` — BOLT framework: pandas CSVs → DuckDB →
  `BookofLifeGeneratorBatch` → `BookofLifeGenerator` → `books.json`.
  Recipe-YAML-driven; `Paragraph` subclasses for new datasets.

Both write the same `data/processed/books.json` shape, so the §4
prediction + metrics code is shared. The class diagram below only
documents the BOLT path.

For "what file to edit when…" see [README.md → What to edit when](README.md#what-to-edit-when).

## Pipeline — `bol_pipeline_simplified.ipynb` (pandas only)

```mermaid
flowchart LR
    subgraph RAW["data/raw/"]
        RawCsv(["NLSY79 CSV (RNUM-headed)"])
        RawCdb(["NLSY79 .cdb codebook"])
        RawDo([".do value-labels file"])
    end

    subgraph LABELS["data/helpers/"]
        BuildLabels["build_value_labels.py"]
        Overrides(["value_labels_overrides.csv"])
        ValueLabels(["value_labels.csv"])
    end

    subgraph META["data/"]
        VarMeta(["variable_metadata.csv"])
    end

    Reshape["reshape_nlsy79.py"]

    subgraph PROCESSED["data/processed/"]
        Person(["nlsy79_person.csv"])
        Year(["nlsy79_year.csv (sparse)"])
        YearFull(["nlsy79_year_full.csv"])
        Targets(["targets.csv"])
        VarIndex(["variable_index.csv"])
    end

    SimRender["inline pandas render in §3 (~50 lines)"]
    Books(["data/processed/books.json"])

    subgraph PREDICT["§4 Predict"]
        LLM(["LLM Prediction"])
        Logreg(["sklearn logistic regression"])
        Preds(["data/processed/predictions.csv"])
    end

    RawCsv --> Reshape
    RawCdb --> Reshape
    RawCdb --> BuildLabels
    RawDo --> BuildLabels
    Overrides -.->|overlay wins| BuildLabels
    BuildLabels --> ValueLabels
    ValueLabels --> Reshape

    VarMeta --> Reshape
    Reshape --> Person & Year & YearFull & Targets & VarIndex

    Person & Year & VarMeta --> SimRender
    SimRender --> Books

    Books & Targets --> LLM
    Person & YearFull & Targets --> Logreg
    LLM --> Preds
    Logreg --> Preds
```

## Pipeline — `bol_pipeline.ipynb` (BOLT framework)

```mermaid
flowchart LR
    subgraph RAW["data/raw/"]
        RawCsv(["NLSY79 CSV (RNUM-headed)"])
        RawCdb(["NLSY79 .cdb codebook"])
        RawDo([".do value-labels file"])
    end

    subgraph LABELS["data/helpers/"]
        BuildLabels["build_value_labels.py"]
        Overrides(["value_labels_overrides.csv"])
        ValueLabels(["value_labels.csv"])
    end

    subgraph META["data/"]
        VarMeta(["variable_metadata.csv"])
    end

    Reshape["reshape_nlsy79.py"]

    subgraph PROCESSED["data/processed/"]
        Person(["nlsy79_person.csv"])
        Year(["nlsy79_year.csv (sparse)"])
        YearFull(["nlsy79_year_full.csv"])
        Targets(["targets.csv"])
        VarIndex(["variable_index.csv"])
    end

    subgraph BOLT["BOLT render"]
        MakeDbYaml>"bolt/recipes/make_db.yaml"]
        TemplateYaml>"bolt/recipes/template.yaml"]
        MakeDb["bolt/serialization/make_db.py"]
        Duck(["bolt/dbs/nlsy79.duckdb"])
        Batch["BookofLifeGeneratorBatch"]
        Gen["BookofLifeGenerator"]
    end

    Books(["data/processed/books.json"])

    subgraph PREDICT["§4 Predict"]
        LLM(["LLM Prediction"])
        Logreg(["sklearn logistic regression"])
        Preds(["data/processed/predictions.csv"])
    end

    RawCsv --> Reshape
    RawCdb --> Reshape
    RawCdb --> BuildLabels
    RawDo --> BuildLabels
    Overrides -.->|overlay wins| BuildLabels
    BuildLabels --> ValueLabels
    ValueLabels --> Reshape

    VarMeta --> Reshape
    Reshape --> Person & Year & YearFull & Targets & VarIndex

    Person & Year --> MakeDb
    MakeDbYaml --> MakeDb
    MakeDb --> Duck
    Duck --> Batch
    TemplateYaml --> Batch
    VarIndex -.->|sentence_templates| Batch
    Batch -->|"INSTANTIATORS map"| Gen
    Gen --> Books

    Books & Targets --> LLM
    Person & YearFull & Targets --> Logreg
    LLM --> Preds
    Logreg --> Preds
```

## Class diagram — BOLT path

Only relevant for `bol_pipeline.ipynb`. The simplified notebook uses a short pandas code block.

```mermaid
classDiagram
    class Paragraph {
        <<abstract>>
        +str dataset_name
        +str caseid
        +bool explicit
        +int order
        +int year
        +int month
        +int day
        +str year_dataset_name
        +str year_month_day
        +get_paragraph_string_tabular(features, sentence_templates)
        +get_paragraph_string_biographic(features, sentence_templates)*
    }

    class PersonParagraph {
        +dict attributes
        +get_paragraph_string_biographic(features, sentence_templates)
    }

    class YearParagraph {
        +int survey_year
        +dict attributes
        +__post_init__()
        +get_paragraph_string_biographic(features, sentence_templates)
    }

    class Recipe {
        +str main_key
        +list datasets
        +list sorting_keys
        +str paragraph_generator
        +get_features(dataset_name)
    }

    class BookofLifeGenerator {
        +str caseid
        +Recipe recipe
        +dict sentence_templates
        +instantiate_paragraphs()
        +sort_paragraphs()
        +write_book(generator_function)
        +generate_book()
    }

    class BookofLifeGeneratorBatch {
        +list caseids
        +Recipe recipe
        +dict sentence_templates
        +list paragraphs_dict_list
        +instantiate_paragraph_dicts()
        +combine_paragraphs(dict_list)
        +write_books()
    }

    Paragraph <|-- PersonParagraph
    Paragraph <|-- YearParagraph
    BookofLifeGenerator *-- Recipe
    BookofLifeGeneratorBatch *-- Recipe
    BookofLifeGenerator o-- Paragraph
    BookofLifeGeneratorBatch o-- Paragraph

    note for BookofLifeGenerator "INSTANTIATORS map (BookofLifeGenerator.py):\n  'nlsy79_person' -> person.get_persons\n  'nlsy79_year'   -> year.get_years\nEach returns Dict[caseid, List[Paragraph]] from DuckDB.\nsort_paragraphs() consumes per-dataset n_spell,\nmin_spell_year, max_spell_year from the recipe."
```

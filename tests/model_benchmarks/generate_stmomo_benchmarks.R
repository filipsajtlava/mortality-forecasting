library(StMoMo)
library(dplyr)
library(here)
library(jsonlite)
library(stringr)

data_dir <- here("tests", "model_benchmarks", "data")
output_dir <- here("tests", "model_benchmarks", "outputs")

data_files <- list.files(data_dir)
available_countries <- unique(str_match(data_files, "^(.+)_")[, 2])

for (country_code in available_countries) {
        deaths_path <- file.path(
            data_dir, 
            paste0(country_code, "_Deaths.csv")
        )
        exposures_path <- file.path(
            data_dir, 
            paste0(country_code, "_Exposures.csv")
        )
    
    deaths_mat <- as.matrix(read.csv(
        deaths_path, row.names = 1, check.names = FALSE
    ))
    expos_mat <- as.matrix(read.csv(
        exposures_path, row.names = 1, check.names = FALSE
    ))

    stmomo_data <- structure(
        list(
            Dxt = deaths_mat,
            Ext = expos_mat,
            ages = as.numeric(rownames(deaths_mat)),
            years = as.numeric(colnames(deaths_mat)),
            type = "central"
        ),
        class = "StMoMoData"
    )

    lc_model <- lc(link = "log", const = "sum")
    fit_lc <- fit(lc_model, data = stmomo_data)
    estimated_parameters <- list(
        ax = as.vector(fit_lc$ax),
        bx = as.vector(fit_lc$bx),
        kt = as.vector(fit_lc$kt)
    )

    write_json(
        estimated_parameters, 
        file.path(output_dir, paste0(country_code, "_estimated_params_poisson.json")), 
        pretty = TRUE
    )
}

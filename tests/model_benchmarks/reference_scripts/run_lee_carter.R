library(demography)
library(here)
library(jsonlite)
library(stringr)


data_dir <- here("tests", "model_benchmarks", "data")
output_dir <- here("tests", "model_benchmarks", "outputs")

data_files <- list.files(data_dir)
available_countries <- unique(str_match(data_files, "^(.+)_")[, 2])

for (country_code in available_countries) {
        mortalities_path <- file.path(
            data_dir, 
            paste0(country_code, "_Mortalities.csv")
        )
        exposures_path <- file.path(
            data_dir, 
            paste0(country_code, "_Exposures.csv")
        )
    
    mortalities_mat <- as.matrix(read.csv(
        mortalities_path, row.names = 1, check.names = FALSE
    ))
    exposures_mat <- as.matrix(read.csv(
        exposures_path, row.names = 1, check.names = FALSE
    ))

    demog_data <- demogdata(
        data = mortalities_mat,
        pop = exposures_mat,
        ages = as.numeric(rownames(mortalities_mat)),
        years = as.numeric(colnames(mortalities_mat)),
        type = "mortality",
        label = "",
        name = "Total"
    )

    fit_lc <- lca(demog_data, adjust = "none")
    estimated_parameters <- list(
        ax = as.vector(fit_lc$ax),
        bx = as.vector(fit_lc$bx),
        kt = as.vector(fit_lc$kt)
    )

    write_json(
        estimated_parameters, 
        file.path(output_dir, paste0(country_code, "_est_params_lc.json")), 
        pretty = TRUE
    )
}

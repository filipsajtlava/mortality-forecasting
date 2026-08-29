library(StMoMo)
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
    exposures_mat <- as.matrix(read.csv(
        exposures_path, row.names = 1, check.names = FALSE
    ))

    stmomo_data <- structure(
        list(
            Dxt = deaths_mat,
            Ext = exposures_mat,
            ages = as.numeric(rownames(deaths_mat)),
            years = as.numeric(colnames(deaths_mat)),
            type = "central"
        ),
        class = "StMoMoData"
    )

    po_model <- lc(link = "log", const = "sum")
    fit_po <- fit(po_model, data = stmomo_data)
    forecasts_analytical <- forecast(
        fit_po, 
        h = 100, 
        kt.method = "mrwd", 
        jumpchoice = "fit",
        level=95
    )
    forecasts_mc <- simulate(
        fit_po, 
        nsim = 100000, 
        h = 100, 
        kt.method = "mrwd", 
        jumpchoice = "fit"
    )

    estimated_parameters <- list(
        parameters = list(
            ax = as.vector(fit_po$ax),
            bx = as.vector(fit_po$bx),
            kt = as.vector(fit_po$kt)
        ),
        forecasts = list(
            analytical = list(
                kt = list(
                    lower = as.numeric(forecasts_analytical$kt.f$lower),
                    point = as.numeric(forecasts_analytical$kt.f$mean),
                    upper = as.numeric(forecasts_analytical$kt.f$upper)
                )
            ),
            monte_carlo = list(
                kt = list(
                    lower = as.numeric(forecasts_mc$kt.f$lower),
                    point = as.numeric(forecasts_mc$kt.f$mean),
                    upper = as.numeric(forecasts_mc$kt.f$upper)
                )
            )
        )
    )

    write_json(
        estimated_parameters, 
        file.path(output_dir, paste0(country_code, "_est_params_poisson.json")), 
        pretty = TRUE
    )
}

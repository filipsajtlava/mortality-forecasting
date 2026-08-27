library(StMoMo)

data_dir <- here("tests", "model_benchmarks", "data")

data(EWMaleData)
write.csv(EWMaleData$Dxt, file.path(data_dir, "GBRTENW_Deaths.csv"))
write.csv(EWMaleData$Ext, file.path(data_dir, "GBRTENW_Exposures.csv"))
write.csv(
    EWMaleData$Dxt / EWMaleData$Ext, 
    file.path(data_dir, "GBRTENW_Mortalities.csv")
)
# Read in level-1 shapefile obtained from https://gadm.org/download_country_v3.html
sf <- readRDS('gadm36_DNK_1_sf.rds')

# Confirm entry 3 is 'North Jutland' region
sf[3,'VARNAME_1']

# Confirm there are 28 polygons for this region
length(sf[3,'geometry'][[1]])

# Write each polygon out to a different csv file
for(i in 1:28) {
    write.csv(sf[3,'geometry'][[1]][[i]],paste0('NorthJutlandPolygon',i,'.csv'))
}

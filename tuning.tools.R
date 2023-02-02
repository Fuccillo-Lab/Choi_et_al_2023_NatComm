require(ggplot2)

individual.neuron.tuning <- function(df, session, neuron, include.groups=FALSE){
    # extract only the row of the dataframe that describes the
    # specified neuron in the specified session.
    dfn1 <- subset(df, Session==session & Neuron==neuron);

    # Discard all the columns that do not contain tuning indices
    last.predictor = ifelse(include.groups, 18, 15)
    dfn1 <- dfn1[,3:last.predictor]
    
    # transpose dataset, taking it from being 1 row x k columns (1
    # neuron x k tuning indices) to being k rows x 1 column. What used
    # to be the column names will now become the names of the rows of
    # the new dataframe.
    dfn1 <- as.data.frame(t(dfn1));

    # rename the only column of the dataframe to be "value"
    colnames(dfn1) <- c('value')

    # create a new column from the row names
    dfn1 <- cbind(predictorName = rownames(dfn1), dfn1);

    # delete the row names
    rownames(dfn1) <- c();

    # set by hand the order of the levels of the factor variable that
    # now is contained in the new column. This is a somewhat technical
    # passage to ensure that below ggplot will preserve the order of
    # the bars in the polar bar plot, and will not try to order them
    # alphabetically.
    dfn1$predictorName <- factor(dfn1$predictorName,
                                 levels=dfn1$predictorName,
                                 ordered=FALSE);
    

    dfn1
}


neuron.tuning.plot <- function(tuning.data, session, neuron, include.groups=FALSE) {

    plot <- ggplot(
        individual.neuron.tuning(tuning.data,session,neuron,include.groups),
        aes(predictorName, value, fill = predictorName)) +
        geom_bar(width = 1, stat = "identity", color = "white") +
        scale_y_continuous(breaks = 0:nlevels(tuning.data$predictorName)) +
        theme_gray() +
        theme(axis.ticks = element_blank(),
              axis.text = element_blank(),
              axis.title = element_blank(),
              axis.line = element_blank())
    plot + coord_polar()
}


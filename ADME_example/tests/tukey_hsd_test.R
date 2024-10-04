library(lmerTest)
library(multcomp)

df <- read.csv("df_scaffold_split.csv")

metric_name <- "prec"
df[[metric_name]] <- as.numeric(df[[metric_name]])
df$method <- factor(df$method)

formula <- as.formula(paste(metric_name, "~ method + (1 | cv_cycle)"))
m <- lmer(formula, data = df)
m_sum <- summary(m)$coefficients
df_residual <- as.integer(m_sum[nrow(m_sum), "df"])

test_out <- multcomp::glht(m, linfct = mcp(method = "Tukey"), df = df_residual)
summary(test_out)
confint(test_out)

old.par <- par(mai=c(1.5,3,1,1))
plot(test_out)


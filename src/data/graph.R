  dag = model2network("[U][C1|U][C2|U][F][G][H][D|F][E|G:H][I|E][B|C2:D:E][K][J|K][L|K][M|K:L][N|J:M][X|C2:J][T|X][Y|C1:T][P|T][Q|P]")
  graphviz.plot(dag)
  
  library(bnlearn)
  bn = custom.fit(dag, list(
    U = list(coef = c("(Intercept)" = 5), sd = 2),
    C1 = list(coef = c("(Intercept)" = 12, "U" = 5), sd = 5.5),
    C2 = list(coef = c("(Intercept)" = 12, "U" = 2), sd = 7.5),
    F = list(coef = c("(Intercept)" = 8), sd = 2),
    G = list(coef = c("(Intercept)" = 3.5), sd = 2),
    H = list(coef = c("(Intercept)" = 3.5), sd = 1.5),
    K = list(coef = c("(Intercept)" = 3), sd = 2.5),
    J = list(coef = c("(Intercept)" = 3, "K" = 2), sd = 1),
    L = list(coef = c("(Intercept)" = 6, "K" = 0.5), sd = 2),
    X = list(coef = c("(Intercept)" = 2, "J" = 2, "C2" = 2), sd = 0.5),
    M = list(coef = c("(Intercept)" = 4, "L" = 4, "K" = 4), sd = 2),
    N = list(coef = c("(Intercept)" = 0, "J" = 0.33, "M" = 0.33), sd = 1),
    T = list(coef = c("(Intercept)" = 1,"X" =0.7), sd = 1),
    Y = list(coef = c("(Intercept)" = 8, "C1" = 2, "T" = 2), sd = 2.5),
    P = list(coef = c("(Intercept)" = 5, "T" = 4), sd = 2),
    Q = list(coef = c("(Intercept)" = 1, "P" = 0.8), sd = 1),
    D = list(coef = c("(Intercept)" = 6, "F"= 1.5), sd = 0.33),
    E = list(coef = c("(Intercept)" = 5, "G" = 0.6, "H"= 0.8), sd = 2.2),
    B = list(coef = c("(Intercept)" = 0, "D"= 2, "C2"= 1, "E"= 1.5), sd = 1),
    I = list(coef = c("(Intercept)" = 0, "E"= 2), sd = 1.5)
  ))
  
  df<- rbn(bn,50)
  
  df1 = subset(df, select = -c(U))
  write.csv(df1,"C:/Users/KIIT/Desktop/Causal/data/small sample/v_8.csv")
  cols= names(df1)

#Data Generation

library(bnlearn)
a = data.frame(U = rnorm(5000, 0, 1),C1 = rep(0, 5000),
               C2 = rep(0, 5000), X = rep(0, 5000),
               T = rep(0, 5000),  Y = rep(0, 5000),
               P = rep(0, 5000),  Q = rep(0, 5000),
               F = rnorm(5000,8,3), G = rnorm(5000,3.5, 2),
               H = rnorm(5000,3, 1),D = rep(0, 5000),
               E = rep(0, 5000),   B = rep(0, 5000),
               I = rep(0, 5000),J = rep(0, 5000),
               L = rep(0, 5000),K = rnorm(5000,5, 2),
               M = rep(0, 5000),N = rep(0, 5000))

a$C1 = 1.5 * a$U + rnorm(5000, 2, 0.5)
a$C2 = 2.5 * a$U + rnorm(5000, 1, 2)
a$J = 2* a$K + rnorm(5000, 3, 1)
a$L = 0.5 * a$K + rnorm(5000, 6, 2)
a$X = 2 * (a$J + a$C2) + rnorm(5000, 2, 0.5)
a$M = 4 * (a$L + a$K) + rnorm(5000, 4, 2)
a$N = 0.33 * (a$J + a$M) + rnorm(5000, 0, 1)
a$T = 0.7 * a$X + rnorm(5000, 0, 1)
a$Y = 2 * (a$C1 + a$T) + rnorm(5000, 8, 2.5)
a$D = 1.5 * a$F + rnorm(5000, 6, 0.33)
a$E = 0.6 * a$G + 0.8 * a$H + rnorm(5000, 5, 2.2)
a$B = 2 * a$D +1.5 * a$C2 + a$E + rnorm(5000, 0, 1)
a$P = 4 * a$T +rnorm(5000, 5, 2)
a$Q = 0.8 * a$P + rnorm(5000, 0, 1)
a$I = 2 * a$E + rnorm(5000,0,1)
# network specification.
dag = model2network("[U][C1|U][C2|U][F][G][H][D|F][E|G:H][I|E][B|C2:D:E][K][J|K][L|K][M|K:L][N|J:M][X|C2:J][T|X][Y|C1:T][P|T][Q|P]")
graphviz.plot(dag)


bn = custom.fit(dag, list(
  U = list(coef = c("(Intercept)" = 8), sd = 2),
  C1 = list(coef = c("(Intercept)" = 2, "U" = 2), sd = 0.5),
  C2 = list(coef = c("(Intercept)" = 2.5, "U" = 1), sd = 1),
  F = list(coef = c("(Intercept)" = 8), sd = 3),
  G = list(coef = c("(Intercept)" = 3.5), sd = 2),
  H = list(coef = c("(Intercept)" = 3), sd = 1),
  K = list(coef = c("(Intercept)" = 3.5), sd = 2),
  J = list(coef = c("(Intercept)" = 3, "K" = 2), sd = 1),
  L = list(coef = c("(Intercept)" = 6, "K" = 0.5), sd = 2),
  X = list(coef = c("(Intercept)" = 2, "J" = 2, "C2" = 2), sd = 0.5),
  M = list(coef = c("(Intercept)" = 4, "L" = 4, "K" = 4), sd = 2),
  N = list(coef = c("(Intercept)" = 0, "J" = 0.33, "M" = 0.33), sd = 1),
  T = list(coef = c("(Intercept)" = 0,"X" =0.7 ), sd = 1),
  Y = list(coef = c("(Intercept)" = 8, "C1" = 2, "T" = 2), sd = 2.5),
  P = list(coef = c("(Intercept)" = 5, "T" = 4), sd = 2),
  Q = list(coef = c("(Intercept)" = 0, "P" = 0.8), sd = 1),
  D = list(coef = c("(Intercept)" = 6, "F"= 1.5), sd = 0.33),
  E = list(coef = c("(Intercept)" = 5, "G" = 0.6, "H"= 0.8), sd = 2.2),
  B = list(coef = c("(Intercept)" = 0, "D"= 2, "C2"= 1.5, "E"= 1), sd = 1),
  I = list(coef = c("(Intercept)" = 0, "E"= 2), sd = 1)
))

df<- rbn(bn,50)

df1 = subset(df, select = -c(U))
write.csv(df1,"C:/Users/KIIT/Desktop/Causal/data/small.csv")
cols= names(df1)

df <- read.csv("C:\\Users\\KIIT\\Desktop\\Causal\\data\\diabetes1.csv")
c <- colnames(df)

df1<- df[df$Gender == "Male",]
df1[,-c(1)] = lapply(df1[,-c(1)], as.factor)
df1$Age = as.numeric(df1$Age)

df1<- df1[df1$Age < 50,]
sapply(df1, class)
df1= df
library(bnlearn)
ls<- pc.stable(df1)
ls<- gs(df1)
ls<- iamb(df1)

ls<- si.hiton.pc(df1)


ls$nodes$class$mb
ls$nodes$N$mb
ls$nodes$M$mb
ls$nodes$E$mb

data.info = bnlearn:::check.data(df1, allow.missing = TRUE)
complete=data.info$complete.nodes
g<-bnlearn:::gs.markov.blanket(x="T", data=df1,nodes=names(df1),whitelist = NULL,
                               blacklist = NULL,test="zf",alpha = 0.05,B=0L,complete=complete,
                               max.sx = ncol(df1))
g
g<-bnlearn:::ia.markov.blanket(x="T", data=df1,nodes=names(df1),whitelist = NULL,
                               blacklist = NULL,test="zf",alpha = 0.05,B=0L,complete=complete,
                               max.sx = ncol(df1))

g
g<-bnlearn:::ia.fdr.markov.blanket(x="T", data=df1,nodes=names(df1),whitelist = NULL,
                                   blacklist = NULL,test="zf",alpha = 0.05,B=0L,
                                   complete=complete,max.sx = ncol(df))

g<-bnlearn:::inter.ia.markov.blanket(x="T", data=df1,nodes=names(df1),whitelist = NULL,
                                     blacklist = NULL,test="zf",alpha = 0.05,B=0L,
                                     complete=complete,max.sx = ncol(df))
g
g<-bnlearn:::fast.ia.markov.blanket(x="T", data=df1,nodes=names(df1),whitelist = NULL,blacklist = NULL,test="zf",alpha = 0.05,B=0L,complete=complete,max.sx = ncol(df))

library(MXM)
g<- mmmb(df1$T , df1 , max_k = 3 , threshold = 0.05, test= "testIndFisher", 
     ncores = 1,)
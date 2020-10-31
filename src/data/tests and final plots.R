
mydata <- read.csv("C:\\Users\\KIIT\\Desktop\\Causal\\supp\\Plot Graph G2 in c1 change Gaussian low sample size_time.csv", header = FALSE)

labels <- c('Baseline','GSS','CMIM+SVR','CMIM+kNNR','CMIM+RFR','MIM+SVR','MIM+kNNR','MIM+RFR','Adaboost+SVR','Adaboost+kNNR','Adaboost+RFR','C4.5+DTR','RTCL')
y<- read.csv(text=c('Methods,t-val'), header = FALSE)
for(i in 2:length(labels)-1) {
  vale <- var.test(unlist(mydata[i]),unlist(mydata[length(labels)]))$p.value
  print(vale)
  if (as.numeric(unlist(vale)) > 0.05){
    val <- t.test(mydata[i],mydata[length(labels)], var.equal = TRUE)$p.value
    y<- rbind(y,c(labels[i],as.numeric(unlist(val))))
    print(val)
  }
  else{
    val <-  t.test(mydata[i],mydata[length(labels)], var.equal = FALSE)$p.value 
    y<- rbind(y,c(labels[i],as.numeric(unlist(val))))
    print(val)
  }
  
}


#PLOT
x <- data.frame(Methods = labels , MSE = unlist(t(mydata)[,1]))
for(i in 2:8){
  x1 <- data.frame(Methods = labels, MSE = unlist(t(mydata)[,i]))
  x <- rbind(x,x1)
}

###################
remove <- c('Adaboost+SVR')#,'C4.5+DTR','CMIM+kNNR','MIM+kNNR')#,)#'CMIM+SVR','MIM+SVR')#,'CMIM+RFR','MIM+RFR')
for (i in remove){
  x <- x[x$Methods != i,] 
}
####################
library(ggplot2)


bp <- ggplot(x, aes(x=Methods, y=MSE, fill=Methods)) + 
  geom_boxplot()+
  labs(y = "Time (in secs)", legend = FALSE)
bp <- bp + theme(axis.text.x = element_text(angle = 90, hjust = 1), axis.title.x=element_blank())
bp<- bp + theme(legend.position = "none")
bp<-bp+ stat_summary(fun = mean, geom = "errorbar", aes(ymax = ..y.., ymin = ..y..),
                     width = 0.76, linetype = "solid",color="white", position = position_dodge())
bp<- bp+theme(plot.margin=unit(c(0.5,0.5,1.8,0.5),"cm"))
bp
pdf("Plot Graph G2 in c1 change Gaussian low sample size_time.pdf")
print(bp)     # Plot 2 ---> in the second page of the PDF
dev.off() 




bp <- ggplot(x, aes(x=Methods, y=MSE, fill=Methods)) + 
  geom_boxplot()+
  labs(title="Plot of Time per Methodologies",x="Methods", y = "Time(in s)")
bp <- bp + theme(axis.text.x = element_text(angle = 50, hjust = 1))

pdf("Time__C1_C2_graph1_highsample(discrete).pdf")
print(bp)     # Plot 2 ---> in the second page of the PDF
dev.off() 

bp


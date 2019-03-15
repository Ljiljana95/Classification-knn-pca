%Zadatak 2
%Napraviti klasifikator koji ce na osnovu prisustva/odsustva odre?enih sastojaka odrediti da li je dati recept
%za pravljenje kolacica (Cookies), peciva (Pastries) ili pice (Pizzas). Prikupljeno je 1738 recepata i za svaki
%je naznaceno prisustvo (1) ili odsustvo (0) svakog od 133 sastojka.

%Ucitavanje podataka
X = load('recepti_train');

%1. Napraviti kNN klasifikator koristeci sva obeležja na sledeci nacin:

%a. Odvojiti deo podataka (15%) za testiranje, pazeci pritom da od svake klase ima dovoljno uzoraka u test skupu.

c = cvpartition(X.lab_train,'HoldOut',0.15);
tr = c.training;
trening = X.data_train((tr == 1),:);
labeletrening = X.lab_train((tr == 1),:);
te = c.test;
test = X.data_train((te == 1),:);
labeletest = X.lab_train((te == 1),:);

%b. Parametar k (broj najbližih suseda) varirati od 1 do 10. Da li možete još nešto da varirate?

%model1 = fitcknn(trening, labeletrening, 'NumNeighbors', 1, 'NSMethod', 'exhaustive', 'Distance', 'minkowski','Exponent',2,'Standardize',1);
%model11 = fitcknn(trening, labeletrening, 'NumNeighbors', 1, 'NSMethod', 'kdtree','BucketSize',50,'Standardize',1);
%model1 = fitcknn(trening, labeletrening, 'NumNeighbors', 1, 'NSMethod', 'exhaustive', 'Distance', 'cosine', 'BreakTies','nearest','Standardize',1);

rng(1) %radi ponovljivosti
model1 = fitcknn(trening, labeletrening, 'NumNeighbors', 1, 'NSMethod', 'exhaustive', 'Distance', 'cosine','Standardize',1);
model2 = fitcknn(trening, labeletrening, 'NumNeighbors', 2, 'NSMethod', 'exhaustive', 'Distance', 'cosine','Standardize',1);
model3 = fitcknn(trening, labeletrening, 'NumNeighbors', 3, 'NSMethod', 'exhaustive', 'Distance', 'cosine','Standardize',1);
model4 = fitcknn(trening, labeletrening, 'NumNeighbors', 4, 'NSMethod', 'exhaustive', 'Distance', 'cosine','Standardize',1);
model5 = fitcknn(trening, labeletrening, 'NumNeighbors', 5, 'NSMethod', 'exhaustive', 'Distance', 'cosine','Standardize',1);
model6 = fitcknn(trening, labeletrening, 'NumNeighbors', 6, 'NSMethod', 'exhaustive', 'Distance', 'cosine','Standardize',1);
model7 = fitcknn(trening, labeletrening, 'NumNeighbors', 7, 'NSMethod', 'exhaustive', 'Distance', 'cosine','Standardize',1);
model8 = fitcknn(trening, labeletrening, 'NumNeighbors', 8, 'NSMethod', 'exhaustive', 'Distance', 'cosine','Standardize',1);
model9 = fitcknn(trening, labeletrening, 'NumNeighbors', 9, 'NSMethod', 'exhaustive', 'Distance', 'cosine','Standardize',1);
model10 = fitcknn(trening, labeletrening, 'NumNeighbors', 10, 'NSMethod', 'exhaustive', 'Distance', 'cosine','Standardize',1);

%Mozemo da variramo rastojanja: Euklidsko rastojanje, ali mogu se koristiti i cityblock, hamming, kosinusno, korelacija...

%c. Izbor modela: krosvalidacijom odabrati najbolji kNN klasifikator

%cv = cvpartition(labeletrening,'HoldOut',0.20);

cvmodel1 = crossval(model1, 'KFold', 10); 
L1 = kfoldLoss(cvmodel1, 'mode', 'average')
% cvmodel1 = crossval(model1, 'Leaveout', 'on'); %Ovaj je najbolji ali je mnogo spor!                            
% L1 = kfoldLoss(cvmodel1, 'mode', 'average')
% cvmodel1 = crossval(model1, 'CVpartition',cv);                                     
% L11 = kfoldLoss(cvmodel1, 'mode', 'average');
% cvmodel1 = crossval(model1, 'Holdout',0.1);                                     
% L111 = kfoldLoss(cvmodel1, 'mode', 'average')
% L1 = min([L1, L11, L111])

cvmodel2 = crossval(model2, 'KFold', 10); 
L2 = kfoldLoss(cvmodel2, 'mode', 'average')
% cvmodel2 = crossval(model2, 'Leaveout', 'on');                                 
% L2 = kfoldLoss(cvmodel2, 'mode', 'average')
% cvmodel2 = crossval(model2, 'CVpartition',cv);                                     
% L22 = kfoldLoss(cvmodel2, 'mode', 'average');
% cvmodel2 = crossval(model2, 'Holdout',0.1);                                     
% L222 = kfoldLoss(cvmodel2, 'mode', 'average')
% L2 = min([L2, L22, L222])

cvmodel3 = crossval(model3, 'KFold', 10); 
L3 = kfoldLoss(cvmodel3, 'mode', 'average')
% cvmodel3 = crossval(model3, 'Leaveout', 'on');                                   
% L3 = kfoldLoss(cvmodel3, 'mode', 'average')
% cvmodel1 = crossval(model3, 'CVpartition',cv);                                     
% L33 = kfoldLoss(cvmodel3, 'mode', 'average');
% cvmodel3 = crossval(model3, 'Holdout',0.1);                                     
% L333 = kfoldLoss(cvmodel3, 'mode', 'average')
% L3 = min([L3, L33, L333])

cvmodel4 = crossval(model4, 'KFold', 10); 
L4 = kfoldLoss(cvmodel4, 'mode', 'average')
% cvmodel4 = crossval(model4, 'Leaveout', 'on');                                     
% L4 = kfoldLoss(cvmodel4, 'mode', 'average')
% cvmodel4 = crossval(model4, 'CVpartition',cv);                                     
% L44 = kfoldLoss(cvmodel4, 'mode', 'average');
% cvmodel4 = crossval(model4, 'Holdout',0.1);                                     
% L444 = kfoldLoss(cvmodel4, 'mode', 'average')
% L4 = min([L4, L44, L444])

cvmodel5 = crossval(model5, 'KFold', 10); 
L5 = kfoldLoss(cvmodel5, 'mode', 'average')
% cvmodel5 = crossval(model5, 'Leaveout', 'on');                                 
% L5 = kfoldLoss(cvmodel5, 'mode', 'average')
% cvmodel5 = crossval(model5, 'CVpartition',cv);                                     
% L55 = kfoldLoss(cvmodel5, 'mode', 'average');
% cvmodel5 = crossval(model5, 'Holdout',0.1);                                     
% L555 = kfoldLoss(cvmodel5, 'mode', 'average')
% L5 = min([L5, L55, L555])

cvmodel6 = crossval(model6, 'KFold', 10); 
L6 = kfoldLoss(cvmodel6, 'mode', 'average')
% cvmodel6 = crossval(model6, 'Leaveout', 'on');                                    
% L6 = kfoldLoss(cvmodel6, 'mode', 'average')
% cvmodel6 = crossval(model6, 'CVpartition',cv);                                     
% L66 = kfoldLoss(cvmodel6, 'mode', 'average');
% cvmodel6 = crossval(model6, 'Holdout',0.1);                                     
% L666 = kfoldLoss(cvmodel6, 'mode', 'average')
% L6 = min([L6, L66, L666])

cvmodel7 = crossval(model7, 'KFold', 10); 
L7 = kfoldLoss(cvmodel7, 'mode', 'average')
% cvmodel7 = crossval(model7, 'Leaveout', 'on');                                  
% L7 = kfoldLoss(cvmodel7, 'mode', 'average')
% cvmodel7 = crossval(model7, 'CVpartition',cv);                                     
% L77 = kfoldLoss(cvmodel7, 'mode', 'average');
% cvmodel7 = crossval(model7, 'Holdout',0.1);                                     
% L777 = kfoldLoss(cvmodel7, 'mode', 'average')
% L7 = min([L7, L77, L777])

cvmodel8 = crossval(model8, 'KFold', 10); 
L8 = kfoldLoss(cvmodel8, 'mode', 'average')
% cvmodel8 = crossval(model8, 'Leaveout', 'on');                                    
% L8 = kfoldLoss(cvmodel8, 'mode', 'average')
% cvmodel8 = crossval(model8, 'CVpartition',cv);                                     
% L88 = kfoldLoss(cvmodel8, 'mode', 'average');
% cvmodel8 = crossval(model8, 'Holdout',0.1);                                     
% L888 = kfoldLoss(cvmodel8, 'mode', 'average')
% L8 = min([L8, L88, L888])

cvmodel9 = crossval(model9, 'KFold', 10); 
L9 = kfoldLoss(cvmodel9, 'mode', 'average')
% cvmodel9 = crossval(model9, 'Leaveout', 'on');                                    
% L9 = kfoldLoss(cvmodel9, 'mode', 'average')
% cvmodel9 = crossval(model9, 'CVpartition',cv);                                    
% L99 = kfoldLoss(cvmodel9, 'mode', 'average');
% cvmodel9 = crossval(model9, 'Holdout',0.1);                                     
% L999 = kfoldLoss(cvmodel9, 'mode', 'average')
% L9 = min([L9, L99, L999])

cvmodel10 = crossval(model10,  'KFold', 10); 
L10 = kfoldLoss(cvmodel10, 'mode', 'average')
% cvmodel10 = crossval(model10, 'Leaveout', 'on');                                    
% L10 = kfoldLoss(cvmodel10, 'mode', 'average')
% cvmodel10 = crossval(model10, 'CVpartition',cv);                                     
% L101 = kfoldLoss(cvmodel10, 'mode', 'average')
% cvmodel10 = crossval(model10, 'Holdout',0.1);                                     
% L102 = kfoldLoss(cvmodel10, 'mode', 'average')
% L10 = min([L10, L101, L102])

%d. Obuciti model nad svim trening podacima, potom nad test podacima izracunati tacnost, preciznost, 
%odziv i specificnost za svaku od klasa, kao i prosecne mere za celokupan klasifikator.

%Najbolji je prvi model i njegova greska iznosi:
Lmin = min([L1,L2,L3,L4,L5,L6,L7,L8,L9,L10])

%  model1 = fitcknn(trening,labeletrening,'NumNeighbors',1,'NSMethod','exhaustive','Distance','cosine','Standardize',1);
%  cvmodel1 = crossval(model1, 'Leaveout', 'on'); %Ovaj daje najmanju gresku ali je spor

label_723 = predict(model1, X.data_train(724, :)); 
label_615 = predict(model1, trening(615, :)); 
label_108t = predict(model1, test(108, :));

predicted_labels = predict(model1, test);
[C,order] = confusionmat(labeletest,predicted_labels')

Kolaci = [C(1,1) C(1,2)+C(1,3); C(2,1)+C(3,1) C(2,2)+C(2,3)+C(3,2)+C(3,3)];
TP1 = C(1,1);
FP1 = C(2,1)+C(3,1);
FN1 = C(1,2)+C(1,3);
TN1 = C(2,2)+C(2,3)+C(3,2)+C(3,3);
Osetljivost1 = TP1/(TP1+FN1); 
Specificnost1 = TN1/(FP1+TN1); 
Tacnost1 = (TP1+TN1)/(TP1+FN1+FP1+TN1); % broj tacno klasifikovanih u celoj populaciji
Preciznost1 = TP1/(TP1+FP1);

Peciva = [C(2,2) C(2,1)+C(2,3); C(1,2)+C(3,2) C(1,1)+C(1,3)+C(3,1)+C(3,3)];
TP2 = C(2,2); 
FN2 = C(2,1)+C(2,3); 
FP2 = C(1,2)+C(3,2); 
TN2 = C(1,1)+C(1,3)+C(3,1)+C(3,3);
Osetljivost2 = TP2/(TP2+FN2);
Specificnost2 = TN2/(FP2+TN2); 
Tacnost2 = (TP2+TN2)/(TP2+FN2+FP2+TN2); 
Preciznost2 = TP2/(TP2+FP2); 

Pice = [C(3,3) C(3,1)+C(3,2); C(1,3)+C(2,3) C(1,1)+C(1,2)+C(2,1)+C(2,2)];
TP3 = C(3,3); 
FN3 = C(3,1)+C(3,2);
FP3 = C(1,3)+C(2,3); 
TN3 = C(1,1)+C(1,2)+C(2,1)+C(2,2);
Osetljivost3 = TP3/(TP3+FN3);
Specificnost3 = TN3/(FP3+TN3); 
Tacnost3 = (TP3+TN3)/(TP3+FN3+FP3+TN3); 
Preciznost3 = TP3/(TP3+FP3);

%prosecne mere za celokupan klasifikator:
Osetljivost = (Osetljivost1 + Osetljivost2 + Osetljivost3)/3;
Specificnost = (Specificnost1 + Specificnost2 + Specificnost3)/3;
Tacnost = (Tacnost1 + Tacnost2 + Tacnost3)/3;
Preciznost = (Preciznost1 + Preciznost2 + Preciznost3)/3;
Prosecne_Mere = [Osetljivost Specificnost Tacnost Preciznost]

%2. Nad citavim setom podataka za svaku od klasa napraviti prosecan obrazac pojavljivanja sastojaka 
%(za svaki sastojak naci koliko se cesto pojavljuje u receptima date klase) i prikazati na 3 odvojena histograma.

kolaci = X.data_train(strcmp(X.lab_train, 'Cookies'),1:end);
mkolaci = mean(kolaci);
peciva = X.data_train(strcmp(X.lab_train, 'Pastries'),1:end);
mpeciva = mean(peciva);
pice = X.data_train(strcmp(X.lab_train, 'Pizzas'),1:end);
mpice = mean(pice);

subplot(3,1,1), stem(mkolaci)
hold on, title('Kolaci')
subplot(3,1,2), stem(mpeciva)
hold on, title('Peciva')
subplot(3,1,3), stem(mpice)
hold on, title('Pice'), xlabel('Prosecan obrazac pojavljivanja sastojaka')


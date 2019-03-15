%1. ZADATAK 1: PCA analiza
%Prvi deo zadatka odnosi se na PCA analizu. Prikupljeni su podaci za 50 država u Americi o
%broju hapšenja na 100 000 stanovnika za ubistva, napade i silovanja. Dat je tako?e i podatak o
%procentu populacije u urbanim zonama.

%Ucitavanje podataka
X = xlsread('USArrests.csv');

%a) Primeniti PCA na dati skup podataka.
[coefs,score,~,~,explained] = pca(zscore(X));

%b) Vizuelizovati podatke pomocu biplot-a (pogledati funkciju u MATLAB-u).
vbls = {'Murder','Assault','UrbanPop','Rape'};
%pca1 = biplot(coefs(:,1:3),'scores',score(:,1:3),'varlabels',vbls);
pca2 = biplot(coefs(:,1:2),'scores',score(:,1:2),'varlabels',vbls);
hold on, title('Biplot data'),xlabel('PC1')
ylabel('PC2'), hold off

%c) Procenat objasnjene varijanse:
PVE1 = explained(1);
PVE2 = explained(1)+explained(2);
PVE3 = explained(1)+explained(2)+explained(3);
PVE4 = explained(1)+explained(2)+explained(3)+explained(4);
PVE = [PVE1 PVE2 PVE3 PVE4];
t = [1:4];
figure
plot(t,PVE),xlabel('Principal Component')
ylabel('Variance Explained (%)')



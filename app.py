from flask import Flask, app,request,render_template
import numpy as np
import pandas as pd
import pickle
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/titanic', methods = ['GET','POST'])
def titanic():
    if request.method == 'POST':
        pclass = request.form.get('pclass')
        name = request.form.get('name')
        sex = request.form.get('sex')
        age = request.form.get('age')
        sibsp = request.form.get('sibsp')
        parch = request.form.get('parch')
        ticket = request.form.get('ticket')
        fare = request.form.get('fare')
        cabin = request.form.get('cabin')
        embarked = request.form.get('embarked')
        input_from_web = {'pclass' : [int(pclass)],'name' : [name],'sex':[sex],'age' : [int(age)],'sibsp' : [int(sibsp)],'parch':[int(parch)],'ticket':[ticket],'fare':[fare],'cabin':[str(cabin)],'embarked':[embarked]}
        input_df = pd.DataFrame(input_from_web)
        #print(input_df)
        #new feature, passenger is man ,woman or child
        def woman_child_or_man(passenger):
            age, sex = passenger
            if age < 16:
                return "child"
            else:
                return dict(male="man", female="woman")[sex]

        input_df["who"] = input_df[["age", "sex"]].apply(woman_child_or_man, axis=1)

        #We will create another feature to see wether a person was an adult male or not."""

        input_df["adult_male"] = input_df.who == "man"

        #We can have another feature with the deck information."""
        input_df["deck"] = input_df.cabin.str[0]

        #Now one more feature can be created, whether the passenger was alone or not. So let's do this."""

        input_df["alone"] = ~(input_df.parch + input_df.sibsp).astype(bool)

        #encoding deck

        dk = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "N":0}
        input_df['deck']=input_df.deck.map(dk)

        # encoding embarked

        input_df['embarked'].value_counts()
        e = {'S':3,'Q':2, 'C':1}
        input_df['embarked']=input_df.embarked.map(e)

        # encoding gender

        genders = {"male": 0, "female": 1}
        input_df['sex'] = input_df['sex'].map(genders)

        #encoding who

        wh = {'child':3,'woman':2, 'man':1}
        input_df['who']=input_df.who.map(wh)

        # Adding New Features"""

        def process_family(parameters):
            x,y=parameters
            # introducing a new feature : the size of families (including the passenger)
            family_size = x+ y + 1
            if (family_size==1):
                return 1 # for singleton
            elif(2<= family_size <= 4 ):
                return 2 #for small family
            else:
                return 3 #for big family

        input_df['FAM_SIZE']= input_df[['parch','sibsp']].apply(process_family, axis=1)

        # to get title from the name.

        titles = set()
        for name in input_df['name']:
            titles.add(name.split(',')[1].split('.')[0].strip())

        #titles #all the salutations present in my dataset.

        len(titles)
        Title_Dictionary = {
            "Capt": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Jonkheer": "Royalty",
            "Don": "Royalty",
            "Sir" : "Royalty",
            "Dr": "Officer",
            "Rev": "Officer",
            "the Countess":"Royalty",
            "Mme": "Mrs",
            "Mlle": "Miss",
            "Ms": "Mrs",
            "Mr" : "Mr",
            "Mrs" : "Mrs",
            "Miss" : "Miss",
            "Master" : "Master",
            "Lady" : "Royalty"
        }

        def get_titles():
        # we extract the title from each name
            input_df['title'] = input_df['name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
            # a map of more aggregated title
            # we map each title
            input_df['title'] = input_df.title.map(Title_Dictionary)
            return input_df

        input_df = get_titles()
        print(input_df)
        Title = input_df['title'].loc[0]
        Title = str(Title)
        print(Title)
        #Now we need to encode these titles. Right now I will use one-hot encoding with this."""
        titles_dummies_dict={'title_Master': 0, 'title_Miss': 0, 'title_Mr': 0, 'title_Mrs': 0, 'title_Officer': 0, 'title_Royalty': 0}
        #print(titles_dummies_dict)
        l='title_'+Title
        titles_dummies_dict[l] = 1
        #print(titles_dummies_dict)
        titles_dummies = pd.DataFrame([titles_dummies_dict])
        input_df = pd.concat([input_df, titles_dummies], axis=1)
        #print(input_df)

        #And finally the Feature that we observed during the visualization.

        def new_fe(parameters):
            p,w=parameters
            if (p==1):
                if (w==1):
                    return 1
                elif (w==2):
                    return 2
                elif (w==3):
                    return 3
            elif (p==2):
                if (w==1):
                    return 4
                elif (w==2):
                    return 5
                elif (w==3):
                    return 6
            elif (p==3):
                if (w==1):
                    return 7
                elif (w==2):
                    return 8
                elif (w==3):
                    return 9

        input_df['pcl_wh']= input_df[['pclass','who']].apply(new_fe, axis=1)

        drop_list=['name','ticket','fare', 'cabin','title']
        input_df = input_df.drop(drop_list, axis=1)
        print(input_df)
        prediction = pickle.load(open('model_titanic.pkl','rb'))
        predicted_value = list(prediction.predict(input_df))
        print(str(predicted_value[0]))
    if str(predicted_value[0]) == '0':
        return render_template('index.html',info = "less chances of survival")
    else:
        return render_template('index.html',info = "high chances of survival")
if __name__==('__main__'):
    app.run(debug=True)
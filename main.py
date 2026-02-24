import os

from dotenv import load_dotenv

# Ładuj .env PRZED importami LangChain/LangSmith (override=True nadpisuje zmienne systemowe)
load_dotenv(override=True)

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langsmith import traceable


@traceable
def main():
    print("Hello from langchain-course!")

    information = """Elon Reeve Musk (wym. /ˈi:lɒn ˈmʌsk/; ur. 28 czerwca 1971 w Pretorii[4]) – południowoafrykański przedsiębiorca, założyciel, współzałożyciel lub finansista przedsiębiorstw SpaceX, Tesla, Neuralink, X.com (część firmy PayPal), The Boring Company[5] oraz xAI. Pochodzi z Republiki Południowej Afryki, mieszka i pracuje w Stanach Zjednoczonych (posiada obywatelstwo południowoafrykańskie, kanadyjskie i amerykańskie). Dyrektor generalny i techniczny w SpaceX, dyrektor generalny i główny architekt w Tesla Inc. W styczniu 2021 został uznany najbogatszym człowiekiem świata przez magazyn „Forbes” i agencję Bloomberg[w innych językach][6][7][8]. Od 28 października 2022 właściciel serwisu X (kiedyś „Twitter”). W okresie od 20 stycznia do 28 maja 2025 szef Departamentu Wydajności Rządu﻿[w innych językach] (Department of Government Efficiency, DOGE) w drugim gabinecie Donalda Trumpa. Na dzień 21 grudnia 2025 roku, według magazynu Forbes, jego majątek szacowany jest na 749 miliardów dolarów amerykańskich (USD)[9].

    Życiorys
    Dzieciństwo i edukacja
    Pochodzi z Południowej Afryki[10]. Urodził się i wychował w stołecznej Pretorii, w białej rodzinie. Jego ojciec, inżynier Errol Musk﻿[w innych językach], urodził się w Południowej Afryce jako syn Brytyjki i Południowoafrykańczyka, natomiast matka, modelka i dietetyczka Maye Musk, ma po ojcu amerykańskie pochodzenie i przyszła na świat w Kanadzie[11][12]. W 1950 wyjechała z rodzicami do Południowej Afryki[11]. Elon ma dwoje młodszego rodzeństwa: brata Kimbala﻿[w innych językach] i siostrę Toscę[12].

    Po rozwodzie rodziców w 1980 mieszkał głównie z ojcem[11]. Kiedy miał 10 lat, dostał pierwszy komputer i nauczył się programować[13] (według innego źródła nauczył się programować w wieku 13 lat)[14]. Dwa lata później sprzedał swój pierwszy program – grę komputerową Blastar za około 500 dolarów[13].

    Jako nastolatek uczęszczał do Pretoria Boys High School, którą ukończył w wieku 17 lat. Krótko później, częściowo z chęci uniknięcia obowiązkowej służby wojskowej w Południowoafrykańskich Siłach Obronnych (SADF), wyemigrował do Kanady, gdzie mieszkała rodzina jego matki[13]. Planował przeniesienie się do Stanów Zjednoczonych[15][16].

    W Kanadzie pracował u kuzyna na farmie w Swift Current, przy czyszczeniu kotłów w tartaku w Kolumbii Brytyjskiej i przy wyrębie lasów. Po dwóch latach przeniósł się do Toronto i pracował w dziale IT w banku, aplikując jednocześnie do Queen’s University. Opuścił Kanadę w 1992 roku po uzyskaniu stypendium na University of Pennsylvania. Tam uzyskał tytuł licencjata w dziedzinie ekonomii na wydziale Wharton Business School, po czym studiował tam jeszcze rok uzyskując tytuł licencjata w dziedzinie fizyki[17]."
    """
    summary_template = """
    Poniżej znajduje się informacja o osoboe Zsumuj ją w krótkim zdaniu.
    dodad dwa ciekawe fakty o osobie.
    {information}
    """


    summary_prompt = PromptTemplate(template=summary_template, input_variables=["information"])
    llm = ChatOllama(model="llama-3.3-70b-versatile", temperature=0)
    chain = summary_prompt | llm
    reponse = chain.invoke(input={"information": information})
    print(reponse.content)



if __name__ == "__main__":
    main()

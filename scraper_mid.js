javascript:(()=> {
    let names = [];
    const players = document.querySelectorAll('.styles__ElementName-sc-52mmxp-5.lhyEpR');
    console.log(players);

    players.forEach(p => {
        names.push(p.innerText);
    });

    console.log('["' + names.join('", "') + '"],');
})();
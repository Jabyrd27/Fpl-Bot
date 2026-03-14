javascript:(()=> {
    var playerData = [
        {  
            'Name': "Andrew",
            'ID': 153978,
            'Players': []
        },
        {
            'Name': "Atticus",
            'ID':154666,
            'Players': []
        },
        {
            'Name': "Matthew",
            'ID': 154679,
            'Players': []
        }
    ];


    const buttons = document.querySelectorAll('.Link-kwe1x-2.hDpGqi');
    console.log(buttons);
    console.log('g');

    for (let i = 0; i < buttons.length; i++) {
        const b = buttons[i];
        const playerName = b.children[2].innerText.trim();
        console.log("B, PlayerName: ", b, playerName);
        b.children[1].click();
        console.log("clicked");

        /*
        setTimeout(() => {
            const players = document.querySelectorAll('.styles__ElementName-sc-52mmxp-5.lhyEpR');

            var player = playerData.find(pd => pd['Name'] == playerName);

            console.log("Player: ", player);

            players.forEach(p => {
                player['Players'].push(p.innerText);
            });
        }, 0);
        */

        setTimeout(() => {
            console.log("going back");
            history.back();
        }, 1500);
    }


    /*console.log('["' + playerNames.join('", "') + '"],');*/

    console.log(playerData);

})

();
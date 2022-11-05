-- Global numbers
select sum(new_cases) as total_cases, sum(new_deaths) as total_deaths, sum(new_deaths) / sum(new_cases) * 100 as death_percentage
from covid.covid_deaths
order by 1, 2;

-- Death count
select continent, sum(new_deaths) as death_count
from covid.covid_deaths
group by continent
order by death_count desc;

-- Countries with highest percentage of population Infected with COVID
select location, population, max(total_cases) as highest_infection_count, max((total_cases / population)) * 100 as percentage_of_population_infected_with_covid
from covid.covid_deaths
group by location, population
order by percentage_of_population_infected_with_covid desc;

select location, population, date, max(total_cases) as highest_infection_count, max((total_cases / population)) * 100 as percentage_of_population_infected_with_covid
from covid.covid_deaths
group by location, population, date
order by percentage_of_population_infected_with_covid desc;